from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .av2_base_dataset import AV2BaseDataset
from utils.numpy import from_numpy

_YAW_LOSS_AGENT_TYPE = [
    0,  # Vehicle
    2,  # Motorcyclist
    3,  # Cyclist
    4,  # Bus
]


class AV2SimplDataset(AV2BaseDataset):
    """
    Simplified dataset built on top of AV2BaseDataset. It uses the shared
    preprocessing utilities and converts the instance-centric scene into the
    SIMPL model specific representation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        history_steps: int = 50,
        future_steps: int = 60,
        max_agents: int = 64,
        max_lanes: int = 128,
        lane_seg_length: float = 15.0,
        num_points_per_lane: int = 10,
        radius: float = 100.0,
        min_distance_threshold: float = 20.0,
        #
        augmentation: bool = False,
        preprocess: bool = True,
        preprocess_dir: str = None,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            hist_steps=history_steps,
            fut_steps=future_steps,
            max_agents=max_agents,
            max_lanes=max_lanes,
            lane_seg_length=lane_seg_length,
            num_points_per_lane=num_points_per_lane,
            radius=radius,
            min_distance_threshold=min_distance_threshold,
        )
        self.augmentation = augmentation
        self.total_steps = history_steps + future_steps
        self.truncate_steps = 2

        self.cache_dir = (
            Path(preprocess_dir) / "av2_simpl" / split
            if preprocess_dir is not None
            else None
        )
        self.preprocess = preprocess
        if self.preprocess:
            if self.cache_dir is None:
                raise ValueError("preprocess_dir must be provided when preprocess=True")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        log_id = self.log_dirs[idx].name
        cache_file = (
            self.cache_dir / f"{log_id}.pt"
            if (self.preprocess and self.cache_dir is not None)
            else None
        )

        if cache_file is not None and cache_file.exists():
            try:
                sample = torch.load(cache_file, map_location="cpu", weights_only=False)
                return self._apply_augmentation(sample)
            except Exception:
                print(f"Warning: failed to load cache file {cache_file}, rebuilding...")

        instance_data = self.get_instance_centric_data(idx)
        sample = self._build_sample_from_instance(instance_data)

        if cache_file is not None:
            torch.save(sample, cache_file)

        return self._apply_augmentation(sample)

    def _build_sample_from_instance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        agent_data = self._process_agent_data(data)
        lane_data = self._process_lane_data(data)
        sample = {
            "scenario_id": data["scenario_id"],
            "city": data["city"],
            "rpe": self._build_rpe(agent_data, lane_data),
            **agent_data,
            **lane_data,
        }
        return sample

    def _process_agent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        agent_pos_local = data["agent_positions"]
        agent_vel_local = data["agent_velocities"]
        agent_ang_local = data["agent_heading_angles"]
        agent_valid_mask = data["agent_valid_masks"]
        agent_last_pos_global = data["agent_last_positions"]
        agent_last_ang_global = data["agent_last_heading_angles"]
        agent_types = data["agent_types"]
        agent_score_types = data["agent_score_types"]

        agent_pos_global = self.transform_local_to_global(
            agent_pos_local,
            agent_last_pos_global[:, None, :],
            agent_last_ang_global[:, None],
        )
        agent_heading_global = self.transform_angle_to_global(
            agent_ang_local,
            agent_last_ang_global[:, None],
        )

        hist_pos = agent_pos_local[:, : self.hist_steps, :]
        hist_disp = np.zeros_like(hist_pos)
        hist_disp[:, 1:, :] = hist_pos[:, 1:, :] - hist_pos[:, :-1, :]

        hist_cos = np.cos(agent_ang_local[:, : self.hist_steps])[..., None]
        hist_sin = np.sin(agent_ang_local[:, : self.hist_steps])[..., None]

        agent_type_onehot = np.eye(7, dtype=np.float32)[np.clip(agent_types, 0, 6)]
        agent_type_hist = np.repeat(
            agent_type_onehot[:, None, :], self.hist_steps, axis=1
        )

        agent_valid_mask_float = agent_valid_mask.astype(np.float32)
        history_mask = agent_valid_mask[:, : self.hist_steps]

        agent_history_feat = np.concatenate(
            [
                hist_disp,
                hist_cos,
                hist_sin,
                agent_vel_local[:, : self.hist_steps, :],
                agent_type_hist,
                agent_valid_mask_float[:, : self.hist_steps, None],
            ],
            axis=-1,
        ).astype(np.float32)

        fut_pos_global = agent_pos_global[:, self.hist_steps :, :]
        fut_ang_global = agent_heading_global[:, self.hist_steps :]
        fut_cossin_global = np.stack(
            [
                np.cos(fut_ang_global),
                np.sin(fut_ang_global),
            ],
            axis=-1,
        ).astype(np.float32)
        fut_valid_mask = agent_valid_mask[:, self.hist_steps :]

        agent_last_rot_global = self._ang_to_rot(agent_last_ang_global)
        yaw_loss_mask = np.isin(agent_types, _YAW_LOSS_AGENT_TYPE)

        return {
            "agent_history": agent_history_feat[:, self.truncate_steps :, :],
            "agent_history_mask": history_mask[
                :, self.truncate_steps : self.hist_steps
            ],
            "agent_future_pos": fut_pos_global,
            "agent_future_ang": fut_cossin_global,
            "agent_future_mask": fut_valid_mask,
            "agent_last_pos": agent_last_pos_global,
            "agent_last_rot": agent_last_rot_global,
            "focal_agent_pos": agent_last_pos_global[0],
            "focal_agent_rot": agent_last_rot_global[0],
            "yaw_loss_mask": yaw_loss_mask,
            "agent_score_types": agent_score_types,
        }

    def _process_lane_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lane_points_local = np.asarray(data["lane_points"], dtype=np.float32)
        lane_anchor_pos = np.asarray(data["lane_anchor_positions"], dtype=np.float32)
        lane_anchor_heading = np.asarray(data["lane_anchor_headings"], dtype=np.float32)
        lane_types = np.asarray(data["lane_types"], dtype=np.int16)
        lane_is_intersect = np.asarray(data["lane_is_intersect"], dtype=np.int16)
        cross_left = np.asarray(data["left_marks"], dtype=np.int16)
        cross_right = np.asarray(data["right_marks"], dtype=np.int16)
        left_nb = np.asarray(data["has_left_nb"], dtype=np.int16)
        right_nb = np.asarray(data["has_right_nb"], dtype=np.int16)

        node_ctrs = (lane_points_local[:, :-1, :] + lane_points_local[:, 1:, :]) / 2.0
        node_vecs = lane_points_local[:, 1:, :] - lane_points_local[:, :-1, :]

        lane_vecs = np.stack(
            [np.cos(lane_anchor_heading), np.sin(lane_anchor_heading)], axis=-1
        ).astype(np.float32)
        lane_ctrs = lane_anchor_pos.astype(np.float32)

        lane_type_onehot = np.eye(3, dtype=np.float32)[np.clip(lane_types, 0, 2)]
        cross_left_onehot = np.eye(3, dtype=np.float32)[np.clip(cross_left, 0, 2)]
        cross_right_onehot = np.eye(3, dtype=np.float32)[np.clip(cross_right, 0, 2)]

        num_lanes = lane_points_local.shape[0]
        num_nodes = node_ctrs.shape[0] * node_ctrs.shape[1]

        return {
            "node_ctrs": node_ctrs.astype(np.float32),
            "node_vecs": node_vecs.astype(np.float32),
            "lane_ctrs": lane_ctrs,
            "lane_vecs": lane_vecs,
            "lane_type": lane_type_onehot,
            "intersect": lane_is_intersect.astype(np.float32),
            "cross_left": cross_left_onehot,
            "cross_right": cross_right_onehot,
            "left": left_nb.astype(np.float32),
            "right": right_nb.astype(np.float32),
            "num_nodes": num_nodes,
            "num_lanes": num_lanes,
        }

    def _ang_to_rot(self, ang: np.ndarray) -> np.ndarray:
        c = np.cos(ang)
        s = np.sin(ang)
        rot = np.stack(
            [
                np.stack([c, -s], axis=-1),
                np.stack([s, c], axis=-1),
            ],
            axis=-2,
        )
        return rot

    def _apply_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def _build_rpe(self, agent_data, lane_data):
        agent_last_pos = torch.from_numpy(agent_data["agent_last_pos"])
        agent_last_ang = torch.from_numpy(agent_data["agent_history"][:, -1, 2:4])

        lane_points = torch.from_numpy(lane_data["lane_ctrs"])
        lane_vectors = torch.from_numpy(lane_data["lane_vecs"])

        scene_points = torch.cat([agent_last_pos, lane_points], dim=0)
        scene_vectors = torch.cat([agent_last_ang, lane_vectors], dim=0)

        dist_matrix = (scene_points.unsqueeze(0) - scene_points.unsqueeze(1)).norm(
            dim=-1
        )
        dist_matrix = dist_matrix / self.radius * 2.0

        def get_cos(v1, v2):
            return (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10)

        def get_sin(v1, v2):
            return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (
                v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10
            )

        heading_vec = scene_vectors.unsqueeze(0)
        heading_vec_t = scene_vectors.unsqueeze(1)
        rel_vec = scene_points.unsqueeze(0) - scene_points.unsqueeze(1)

        rpe = torch.stack(
            [
                get_cos(heading_vec, heading_vec_t),
                get_sin(heading_vec, heading_vec_t),
                get_cos(heading_vec, rel_vec),
                get_sin(heading_vec, rel_vec),
                dist_matrix,
            ],
            dim=0,
        )

        return rpe.float()

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        B = len(batch)
        batch = from_numpy(batch)

        new_batch = {key: [x[key] for x in batch] for key in batch[0].keys()}

        b_agent_dict = self._actor_gather(B, new_batch)
        lane_feats, lane_masks, lane_ctrs, lane_vecs = self._lane_gather(B, new_batch)

        return {
            **b_agent_dict,
            "lane_ctrs": lane_ctrs,
            "lane_vecs": lane_vecs,
            "lane_feats": lane_feats,
            "lane_masks": lane_masks,
            "rpe": new_batch["rpe"],
            "scenario_id": new_batch["scenario_id"],
            "city": new_batch["city"],
        }

    def _actor_gather(self, batch_size, batch):
        num_agents = [x.shape[0] for x in batch["agent_history"]]
        max_agents = max(num_agents)

        b_agent_history = torch.zeros(
            [batch_size, max_agents, self.hist_steps - self.truncate_steps, 14]
        )
        b_agent_history_mask = torch.zeros(
            [batch_size, max_agents, self.hist_steps - self.truncate_steps],
            dtype=torch.bool,
        )
        b_agent_future_pos = torch.zeros([batch_size, max_agents, self.fut_steps, 2])
        b_agent_future_ang = torch.zeros([batch_size, max_agents, self.fut_steps, 2])
        b_agent_future_mask = torch.zeros(
            [batch_size, max_agents, self.fut_steps], dtype=torch.bool
        )
        b_agent_last_pos = torch.zeros([batch_size, max_agents, 2])
        b_agent_last_rot = torch.zeros([batch_size, max_agents, 2, 2])
        b_yaw_loss_mask = torch.zeros([batch_size, max_agents], dtype=torch.bool)
        for i in range(batch_size):
            b_agent_history[i, : num_agents[i]] = batch["agent_history"][i]
            b_agent_history_mask[i, : num_agents[i]] = batch["agent_history_mask"][i]
            b_agent_future_pos[i, : num_agents[i]] = batch["agent_future_pos"][i]
            b_agent_future_ang[i, : num_agents[i]] = batch["agent_future_ang"][i]
            b_agent_future_mask[i, : num_agents[i]] = batch["agent_future_mask"][i]
            b_agent_last_pos[i, : num_agents[i]] = batch["agent_last_pos"][i]
            b_agent_last_rot[i, : num_agents[i]] = batch["agent_last_rot"][i]
            b_yaw_loss_mask[i, : num_agents[i]] = batch["yaw_loss_mask"][i]

        return {
            "agent_history": b_agent_history,
            "agent_history_mask": b_agent_history_mask,
            "agent_future_pos": b_agent_future_pos,
            "agent_future_ang": b_agent_future_ang,
            "agent_future_mask": b_agent_future_mask,
            "agent_last_pos": b_agent_last_pos,
            "agent_last_rot": b_agent_last_rot,
            "yaw_loss_mask": b_yaw_loss_mask,
            "focal_agent_pos": torch.stack(batch["focal_agent_pos"], dim=0),
            "focal_agent_rot": torch.stack(batch["focal_agent_rot"], dim=0),
            "agent_score_types": batch["agent_score_types"],
        }

    def _lane_gather(self, batch_size, graphs):
        num_lanes = graphs["num_lanes"]
        max_lanes = max(num_lanes) if num_lanes else 0

        lane_feats = torch.zeros([batch_size, max_lanes, self.num_lane_nodes, 16])
        lane_masks = torch.zeros([batch_size, max_lanes], dtype=torch.bool)

        lane_ctrs = torch.zeros([batch_size, max_lanes, 2])
        lane_vecs = torch.zeros([batch_size, max_lanes, 2])

        num_nodes_per_lane = self.num_lane_nodes if max_lanes > 0 else 0

        for i in range(batch_size):
            if num_lanes[i] == 0:
                continue

            feat = torch.concat(
                [
                    graphs["node_ctrs"][i],
                    graphs["node_vecs"][i],
                    graphs["intersect"][i][:, None, None].expand(
                        -1, num_nodes_per_lane, -1
                    ),
                    graphs["lane_type"][i][:, None, :].expand(
                        -1, num_nodes_per_lane, -1
                    ),
                    graphs["cross_left"][i][:, None, :].expand(
                        -1, num_nodes_per_lane, -1
                    ),
                    graphs["cross_right"][i][:, None, :].expand(
                        -1, num_nodes_per_lane, -1
                    ),
                    graphs["left"][i][:, None, None].expand(-1, num_nodes_per_lane, -1),
                    graphs["right"][i][:, None, None].expand(
                        -1, num_nodes_per_lane, -1
                    ),
                ],
                dim=-1,
            )

            lane_feats[i, : num_lanes[i]] = feat
            lane_masks[i, : num_lanes[i]] = True
            lane_ctrs[i, : num_lanes[i]] = graphs["lane_ctrs"][i]
            lane_vecs[i, : num_lanes[i]] = graphs["lane_vecs"][i]

        return lane_feats, lane_masks, lane_ctrs, lane_vecs
