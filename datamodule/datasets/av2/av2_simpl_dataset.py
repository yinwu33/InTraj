import os
import math
import random
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import LineString

# AV2 API Imports
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)


from utils.numpy import from_numpy


_AGENT_TYPE_MAP = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.MOTORCYCLIST: 2,
    ObjectType.CYCLIST: 3,
    ObjectType.BUS: 4,
    ObjectType.UNKNOWN: 5,
}

_LANE_TYPE_MAP = {
    LaneType.VEHICLE: 0,
    LaneType.BIKE: 1,
    LaneType.BUS: 2,
}

_LANE_MARK_TYPE = {
    LaneMarkType.DASH_SOLID_YELLOW: 0,
    LaneMarkType.DASH_SOLID_WHITE: 0,
    LaneMarkType.DASHED_WHITE: 0,
    LaneMarkType.DASHED_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_WHITE: 0,
    LaneMarkType.DOUBLE_SOLID_YELLOW: 1,
    LaneMarkType.DOUBLE_SOLID_WHITE: 1,
    LaneMarkType.SOLID_YELLOW: 1,
    LaneMarkType.SOLID_WHITE: 1,
    LaneMarkType.SOLID_DASH_WHITE: 1,
    LaneMarkType.SOLID_DASH_YELLOW: 1,
    LaneMarkType.SOLID_BLUE: 1,
}

_YAW_LOSS_AGENT_TYPE = {0, 2, 3, 4}


class AV2SimplDataset(Dataset):
    """
    A unified dataset class for Argoverse 2 Motion Forecasting.
    Handles preprocessing from raw logs, caching, and loading for training.
    """

    def __init__(
        self,
        data_root: str,
        preprocess_dir: str,
        split: str = "train",
        history_steps: int = 50,
        future_steps: int = 60,
        radius: float = 100.0,
        min_distance_threshold: float = 20.0,
        augmentation: bool = False,
        preprocess: bool = True,
    ):
        """
        Args:
            raw_data_path: Path to the raw AV2 dataset (folders containing log_map_archive and scenarios).
            cached_data_path: Path where processed .pt files will be saved/loaded.
            split: 'train', 'val', or 'test'.
            history_steps: Number of history frames (default 50).
            future_steps: Number of future frames to predict (default 60).
            min_distance_threshold: Filter out actors too far from the map (default 20.0).
            augmentation: Whether to apply data augmentation (random flips).
            preprocess: If True, ignores existing cache and re-processes data.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.preprocess = preprocess

        self.history_steps = history_steps
        self.future_steps = future_steps
        self.min_dist_threshold = min_distance_threshold
        self.augmentation = augmentation
        self.total_steps = history_steps + future_steps
        self.truncate_steps = 2

        self.radius = radius
        self.num_lane_seg = 15.0  # Approximated lane segment length
        self.num_lane_nodes = 10  # Number of nodes per segment

        self.log_dirs = sorted((self.data_root / split).glob("*"))
        self.cache_dir = Path(preprocess_dir) / "av2_simpl" / split
        if self.preprocess:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.log_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a scenario, processes it if necessary (and saves to cache),
        applies augmentation, and formats input.
        """
        log_dir = self.log_dirs[idx]
        log_id = log_dir.name
        cache_file = self.cache_dir / f"{log_id}.pt" if self.preprocess else None
        # raw_path = self.data_root / log_id

        if cache_file is not None and cache_file.exists():
            try:
                sample = torch.load(cache_file, map_location="cpu", weights_only=False)
                sample = self._apply_augmentation(sample)
                return sample
            except Exception:
                print(f"Warning: failed to load cache file {cache_file}, rebuilding...")

        sample = self._build_from_raw(log_id, log_dir)

        if cache_file is not None:
            torch.save(sample, cache_file)

        sample = self._apply_augmentation(sample)
        return sample

    def _build_from_raw(self, log_id: str, log_dir: Path) -> Optional[Dict]:
        json_file = log_dir / f"log_map_archive_{log_id}.json"
        parquet_file = log_dir / f"scenario_{log_id}.parquet"
        if not json_file.exists():
            raise FileNotFoundError(f"{json_file} not found")
        if not parquet_file.exists():
            raise FileNotFoundError(f"{parquet_file} not found")

        # Load raw AV2 objects
        static_map = ArgoverseStaticMap.from_json(json_file)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_file)

        # 1. Extract Trajectories
        # trajs_* shapes: [N, 110, ...]
        agent_data_dict = self._extract_agent_data(scenario, static_map)

        # 2. Extract Lane Graph (Vector Map)
        lane_graph_dict = self._extract_lane_data(
            static_map,
            agent_data_dict["focal_agent_point"],
            agent_data_dict["focal_agent_rotation"],
        )

        rpe = self._build_rpe(agent_data_dict, lane_graph_dict)

        sample = {"scenario_id": log_id, "city": scenario.city_name, "rpe": rpe}
        sample.update(agent_data_dict)
        sample.update(lane_graph_dict)
        return sample

    def _extract_agent_data(
        self, scenario: ArgoverseScenario, static_map: ArgoverseStaticMap
    ):
        """
        Extracts, filters, normalizes, and pads trajectories.
        """
        # Identify Track Indices
        focal_idx, ego_idx = None, None
        scored_idcs, unscored_idcs, fragment_idcs = [], [], []

        for idx, track in enumerate(scenario.tracks):
            if (
                track.track_id == scenario.focal_track_id
                and track.category == TrackCategory.FOCAL_TRACK
            ):
                focal_idx = idx
            elif track.track_id == "AV":
                ego_idx = idx
            elif track.category == TrackCategory.SCORED_TRACK:
                scored_idcs.append(idx)
            elif track.category == TrackCategory.UNSCORED_TRACK:
                unscored_idcs.append(idx)
            elif track.category == TrackCategory.TRACK_FRAGMENT:
                fragment_idcs.append(idx)

        # Enforce existence of AV and Focal
        if focal_idx is None or ego_idx is None:
            raise ValueError(
                f"Focal or ego track not found in scenario {scenario.scenario_id}"
            )

        # Sorting order: Focal -> AV -> Scored -> Unscored -> Fragments
        sorted_indices = (
            [focal_idx, ego_idx] + scored_idcs + unscored_idcs + fragment_idcs
        )
        sorted_categories = (
            ["focal", "av"]
            + ["score"] * len(scored_idcs)
            + ["unscore"] * len(unscored_idcs)
            + ["frag"] * len(fragment_idcs)
        )
        sorted_track_indices = [scenario.tracks[idx].track_id for idx in sorted_indices]

        # Timestamps
        if self.split == "test":
            # Test set only has obs
            ts_frame_indices = np.arange(0, self.history_steps)
        else:
            ts_frame_indices = np.arange(0, self.total_steps)

        last_history_index = self.history_steps - 1  # 49

        # Pre-fetch map points for distance filtering
        lane_points = []
        for lane_id, lane in static_map.vector_lane_segments.items():
            lane_points.append(static_map.get_lane_segment_centerline(lane_id)[:, 0:2])
        lane_points = np.concatenate(lane_points, axis=0)  # [N_map, 2]
        lane_points = np.expand_dims(lane_points, axis=0)  # [1, N_map, 2]

        # Containers
        agent_pos_global_list = []
        agent_ang_global_list = []
        agent_vel_global_list = []
        agent_last_pos_global_list = []
        agent_last_rot_global_list = []
        agent_score_types_list = []
        agent_type_list = []
        valid_mask_list = []
        yaw_loss_mask_list = []

        # * get original scene-centric agent trajectories
        for i, track_idx in enumerate(sorted_indices):
            track = scenario.tracks[track_idx]
            agent_score_type = sorted_categories[i]

            # Extract raw states
            i_agent_ts = np.array(
                [x.timestep for x in track.object_states], dtype=np.int16
            )
            i_agent_pos = np.array([x.position for x in track.object_states])  # [T, 2]
            i_agent_ang = np.array([x.heading for x in track.object_states])  # [T, ]
            i_agent_vel = np.array([x.velocity for x in track.object_states])  # [T, 2]

            # Filter: Skip if strictly future or not present at observation time
            if (
                i_agent_ts[0] > last_history_index
                or last_history_index not in i_agent_ts
            ):
                continue

            # Distance Filter (Skip irrelevant background actors)
            # Only check for non-scored/non-focal/non-av tracks
            if sorted_categories[i] in ["unscore", "frag"]:
                # Check distance of observed points to map
                history_mask = i_agent_ts <= last_history_index
                agent_history_points = np.expand_dims(
                    i_agent_pos[history_mask], axis=1
                )  # [T_obs, 1, 2]
                dist = np.linalg.norm(agent_history_points - lane_points, axis=-1)
                if np.min(dist) > self.min_dist_threshold:
                    continue

            idx_at_curr = np.where(i_agent_ts == last_history_index)[0][0]

            i_agent_last_pos = i_agent_pos[idx_at_curr]  # [2]
            i_agent_last_ang = i_agent_ang[idx_at_curr]  # Scalar
            i_agent_last_rot = np.array(
                [
                    [np.cos(i_agent_last_ang), -np.sin(i_agent_last_ang)],
                    [np.sin(i_agent_last_ang), np.cos(i_agent_last_ang)],
                ]
            )  # [2, 2]


            agent_last_pos_global_list.append(i_agent_last_pos)
            agent_last_rot_global_list.append(i_agent_last_rot)

            # --- Padding ---
            # Create full length arrays filled with nan/zeros
            # Flags (1 if present)
            valid_mask = np.zeros(self.total_steps, dtype=np.bool)
            has_ts_mask = np.isin(i_agent_ts, ts_frame_indices)
            # Map valid timestamps to indices in the fixed array
            mapped_indices = i_agent_ts[has_ts_mask]
            valid_mask[mapped_indices] = True

            # Object Type One-Hot (7 classes)
            # Vehicle, Pedestrian, Motorcyclist, Cyclist, Bus, Unknown, Static
            agent_type_1hot = np.zeros(7)
            type_idx = _AGENT_TYPE_MAP.get(
                track.object_type, 6
            )  # Default to 6 (Static)
            agent_type_1hot[type_idx] = 1
            yaw_loss_mask_list.append(1 if type_idx in _YAW_LOSS_AGENT_TYPE else 0)

            agent_type = np.zeros((self.total_steps, 7))
            agent_type[mapped_indices] = agent_type_1hot

            # Position & Angle (Nearest Neighbor Padding)
            agent_pos_pad = np.full((self.total_steps, 2), np.nan)
            agent_pos_pad[mapped_indices] = i_agent_pos[has_ts_mask]
            agent_pos_pad = self._padding_nearest_neighbor(agent_pos_pad)

            agent_ang_pad = np.full(self.total_steps, np.nan)
            agent_ang_pad[mapped_indices] = i_agent_ang[has_ts_mask]
            agent_ang_pad = self._padding_nearest_neighbor(agent_ang_pad)

            # Velocity (Zero Padding)
            agent_vel_pad = np.zeros((self.total_steps, 2))
            agent_vel_pad[mapped_indices] = i_agent_vel[has_ts_mask]

            # Append
            agent_pos_global_list.append(agent_pos_pad)
            agent_ang_global_list.append(agent_ang_pad)
            agent_vel_global_list.append(agent_vel_pad)
            agent_type_list.append(agent_type)
            valid_mask_list.append(valid_mask)
            agent_score_types_list.append(agent_score_type)

        agent_pos_global_np = np.array(
            agent_pos_global_list, dtype=np.float32
        )  # [N, 110, 2]
        agent_ang_global_np = np.array(
            agent_ang_global_list, dtype=np.float32
        )  # [N, 110]
        agent_cs_global_np = np.stack(
            [np.cos(agent_ang_global_np), np.sin(agent_ang_global_np)], axis=-1
        )  # [N, 110, 2]
        agent_vel_global_np = np.array(
            agent_vel_global_list, dtype=np.float32
        )  # [N, 110, 2]
        agent_last_pos_global_np = np.array(
            agent_last_pos_global_list, dtype=np.float32
        )  # [N, 2]
        agent_last_rot_global_np = np.array(
            agent_last_rot_global_list, dtype=np.float32
        )  # [N, 2, 2]
        agent_type_np = np.array(agent_type_list, dtype=np.int16)
        valid_mask_np = np.array(valid_mask_list, dtype=np.int16)
        yaw_loss_np = np.array(yaw_loss_mask_list, dtype=np.int16)

        # * Now global scene-centric to agent-centric (based on each agent's last pos/rot)
        _t = agent_last_pos_global_np[:, None, :]  # [N, 1, 2]
        _R = agent_last_rot_global_np
        agent_pos_local_np = np.matmul(agent_pos_global_np - _t, _R)
        agent_vel_local_np = np.matmul(agent_vel_global_np, _R)
        ref_yaw_np = np.arctan2(_R[:, 1, 0], _R[:, 0, 0])
        agent_ang_local_np = agent_ang_global_np - ref_yaw_np[:, None]
        agent_ang_local_np = (agent_ang_local_np + np.pi) % (2 * np.pi) - np.pi

        agent_history_pos_local_np = agent_pos_local_np[:, : self.history_steps, :]
        agent_history_displacement = np.zeros_like(agent_history_pos_local_np)
        agent_history_displacement[:, 1:, :] = (
            agent_history_pos_local_np[:, 1:, :] - agent_history_pos_local_np[:, :-1, :]
        )

        agent_history_feat = np.concatenate(
            [
                agent_history_displacement,  # [N, 50, 2]
                np.cos(agent_ang_local_np)[
                    :, : self.history_steps, np.newaxis
                ],  # [N, 50, 1]
                np.sin(agent_ang_local_np)[
                    :, : self.history_steps, np.newaxis
                ],  # [N, 50, 1]
                agent_vel_global_np[:, : self.history_steps, :],  # [N, 50, 2]
                agent_type_np[:, : self.history_steps, :],  # [N, 50, 7]
                valid_mask_np[:, : self.history_steps, np.newaxis],  # [N, 50,
            ],
            axis=-1,
        )  # [N, 50, 14]

        # Stack into arrays
        return {
            "agent_history": agent_history_feat[
                :, self.truncate_steps :, :
            ],  # [N, 50-2, 14]
            "agent_history_mask": valid_mask_np[
                :, self.truncate_steps : self.history_steps
            ],  # [N, 50-2]
            "agent_future_pos": agent_pos_global_np[
                :, self.history_steps :, :
            ],  # [N, 60, 2]
            "agent_future_ang": agent_cs_global_np[
                :, self.history_steps :, :
            ],  # [N, 60, 2]
            "agent_future_mask": valid_mask_np[:, self.history_steps :],  # [N, 60]
            # "agent_last_pos": agent_last_pos,  # [N, 2]
            "agent_last_pos": agent_last_pos_global_np,  # [N, 2]
            "agent_last_rot": agent_last_rot_global_np,  # [N, 2, 2]
            "focal_agent_point": agent_last_pos_global_np[0],  # [2, ]
            "focal_agent_rotation": agent_last_rot_global_np[0],  # [2, 2]
            "yaw_loss_mask": yaw_loss_np,  # [N,]
            "agent_score_types": agent_score_types_list,
        }

    def _padding_nearest_neighbor(self, traj):
        """Fills NaNs/Nones with the nearest valid value (forward then backward)."""
        # Note: Input assumes NaNs for float arrays.
        n = len(traj)
        # Forward pass
        buff = None
        for i in range(n):
            if not np.isnan(traj[i]).any():
                buff = traj[i]
            elif buff is not None:
                traj[i] = buff

        # Backward pass
        buff = None
        for i in reversed(range(n)):
            if not np.isnan(traj[i]).any():
                buff = traj[i]
            elif buff is not None:
                traj[i] = buff

        # Handle case where everything is NaN (shouldn't happen due to logic)
        if np.isnan(traj).any():
            traj[np.isnan(traj)] = 0.0

        return traj

    def _extract_lane_data(
        self,
        static_map: ArgoverseStaticMap,
        orig: np.ndarray,
        rot: np.ndarray,
    ):
        """
        Discretizes lane segments and extracts topological features.
        """
        # * 1. collect lane and do segment and interpolation
        lane_segments_global_np = []
        lane_types = []
        lane_is_intersect = []
        cross_left, cross_right = [], []
        left_nb, right_nb = [], []
        for lane_id, lane in static_map.vector_lane_segments.items():
            # Get Centerline
            centerline_raw = static_map.get_lane_segment_centerline(lane_id)[
                :, 0:2
            ]  # [N, 2]
            centerline_ls = LineString(centerline_raw)

            # Determine number of sub-segments
            num_segs = max(int(np.floor(centerline_ls.length / self.num_lane_seg)), 1)
            ds = centerline_ls.length / num_segs

            for i in range(num_segs):
                # Interpolate points for this sub-segment
                sub_lower_bound = i * ds
                sub_upper_bound = (i + 1) * ds
                num_sub_nodes = self.num_lane_nodes

                centerline_points = [
                    centerline_ls.interpolate(s)
                    for s in np.linspace(
                        sub_lower_bound, sub_upper_bound, num_sub_nodes + 1
                    )
                ]
                centerline_points = np.array(
                    LineString(centerline_points).coords
                )  # [N+1, 2], [11, 2]

                lane_segments_global_np.append(centerline_points)

                # lane type
                lane_type = np.zeros(3)
                lane_type[_LANE_TYPE_MAP[lane.lane_type]] = 1
                lane_types.append(lane_type)

                # lane intersection
                lane_is_intersect.append(1.0 if lane.is_intersection else 0.0)

                # lane mark type:
                cross_left_1hot = np.zeros(3)
                cross_right_1hot = np.zeros(3)
                cross_left_1hot[_LANE_MARK_TYPE.get(lane.left_mark_type, 2)] = 1
                cross_right_1hot[_LANE_MARK_TYPE.get(lane.right_mark_type, 2)] = 1

                cross_left.append(cross_left_1hot)
                cross_right.append(cross_right_1hot)
                # neighbors
                left_nb.append(1 if lane.left_neighbor_id else 0)
                right_nb.append(1 if lane.right_neighbor_id else 0)

        lane_segments_global_np = np.array(
            lane_segments_global_np
        )  # [M, 11, 2], global coords
        lane_types_np = np.array(lane_types).astype(np.int16)  # [M, 3]
        lane_is_intersect_np = np.array(lane_is_intersect).astype(np.int16)  # [M, ]
        cross_left_np = np.array(cross_left).astype(np.int16)  # [M, ]
        cross_right_np = np.array(cross_right).astype(np.int16)  # [M, ]
        left_nb_np = np.array(left_nb).astype(np.int16)  # [M, ]
        right_nb_np = np.array(right_nb).astype(np.int16)  # [M, ]

        # * 2. get anchor of each lane segment
        lane_anchor_pos_global = np.mean(lane_segments_global_np, axis=1)  # [M, 2]
        lane_anchor_vec_global = (
            lane_segments_global_np[:, -1, :] - lane_segments_global_np[:, 0, :]
        )  # [M, 2]
        lane_anchor_vec_global = lane_anchor_vec_global / (
            np.linalg.norm(lane_anchor_vec_global, axis=1, keepdims=True) + 1e-6
        )  # [M, 2]
        lane_anchor_rot_global = np.array(
            [
                [lane_anchor_vec_global[:, 0], -lane_anchor_vec_global[:, 1]],
                [lane_anchor_vec_global[:, 1], lane_anchor_vec_global[:, 0]],
            ],
        )  # [2, 2, M]
        lane_anchor_rot_global = np.transpose(
            lane_anchor_rot_global, (2, 0, 1)
        )  # [M, 2, 2]

        # * 3. transform each lane segment to anchor frame
        lane_segments_to_anchor = (
            lane_segments_global_np - lane_anchor_pos_global[:, None, :]
        ) @ lane_anchor_rot_global  # [M, 11, 2]

        # * 4. process each lane segment to get features
        lane_ctrs_to_anchor = (
            lane_segments_to_anchor[:, :-1, :] + lane_segments_to_anchor[:, 1:, :]
        ) / 2.0  # [M, 10, 2]
        lane_vecs_to_anchor = (
            lane_segments_to_anchor[:, 1:, :] - lane_segments_to_anchor[:, :-1, :]
        )  # [M, 10, 2]

        # Stack Dictionary
        lane_graph_dict = {
            "node_ctrs": lane_ctrs_to_anchor.astype(np.float32),
            "node_vecs": lane_vecs_to_anchor.astype(np.float32),
            "lane_ctrs": lane_anchor_pos_global.astype(np.float32),
            "lane_vecs": lane_anchor_vec_global.astype(np.float32),
            "lane_type": lane_types_np.astype(np.int16),  # [N, 3]
            "intersect": lane_is_intersect_np.astype(np.int16),  # [N, ]
            "cross_left": cross_left_np.astype(
                np.int16
            ),  # [N, 10, 3] one-host: [Crossable, Not-Crossable, Unknown]
            "cross_right": cross_right_np.astype(np.int16),  # [N, 10, 3]
            "left": left_nb_np.astype(np.int16),  # [N, 10]
            "right": right_nb_np.astype(np.int16),  # [N, 10]
            "num_nodes": lane_ctrs_to_anchor.shape[0] * lane_ctrs_to_anchor.shape[1],
            "num_lanes": lane_ctrs_to_anchor.shape[0],
        }

        return lane_graph_dict

    def _apply_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
        # TODO
        """Applies random vertical flip with 30% probability."""
        # Deep copy needed if we are modifying in place and caching is in memory
        # But here we load fresh from disk each time, so partial copy is okay.

        # Only flip if enabled and coin toss passes
        if not (self.augmentation and random.random() < 0.3):
            return data

        # Flip Y coordinates
        # Trajectories
        data["TRAJS"]["trajs_ctrs"][..., 1] *= -1
        data["TRAJS"]["trajs_vecs"][..., 1] *= -1
        data["TRAJS"]["trajs_pos"][..., 1] *= -1
        data["TRAJS"]["trajs_ang"] *= -1  # Angle flip
        data["TRAJS"]["trajs_vel"][..., 1] *= -1

        # Lane Graph
        data["LANE_GRAPH"]["lane_ctrs"][..., 1] *= -1
        data["LANE_GRAPH"]["lane_vecs"][..., 1] *= -1
        data["LANE_GRAPH"]["node_ctrs"][..., 1] *= -1
        data["LANE_GRAPH"]["node_vecs"][..., 1] *= -1

        # Swap Left/Right Neighbor attributes
        lg = data["LANE_GRAPH"]
        lg["left"], lg["right"] = lg["right"], lg["left"]

        return data

    def _build_rpe(self, agent_data, lane_data):
        agent_last_pos = agent_data["agent_last_pos"]  # [N, 2]
        agent_last_ang = agent_data["agent_history"][:, -1, 2:4]  # [N, 2]

        lane_points = lane_data["lane_ctrs"]  # [M, 2]
        lane_vectors = lane_data["lane_vecs"]  # [M, 2]

        scene_points = torch.cat(
            [torch.from_numpy(agent_last_pos), torch.from_numpy(lane_points)],
            dim=0,
        )  # [N+M, 2]
        scene_vectors = torch.cat(
            [torch.from_numpy(agent_last_ang), torch.from_numpy(lane_vectors)],
            dim=0,
        )  # [N+M, 2]

        # distance matrix [N+M, N+M]
        dist_matrix = (scene_points.unsqueeze(0) - scene_points.unsqueeze(1)).norm(
            dim=-1
        )
        dist_matrix = dist_matrix / self.radius * 2.0  # scale [0, 100] to [0, 2]

        # angle diff
        def get_cos(v1, v2):
            return (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10)

        def get_sin(v1, v2):
            return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (
                v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-10
            )

        heading_vec = scene_vectors.unsqueeze(0)  # [1, N+M, 2]
        heading_vec_t = scene_vectors.unsqueeze(1)  # [N+M, 1, 2]
        rel_vec = scene_points.unsqueeze(0) - scene_points.unsqueeze(1)  # [N+M, N+M, 2]

        rpe = torch.stack(
            [
                get_cos(heading_vec, heading_vec_t),  # [N+M, N+M]
                get_sin(heading_vec, heading_vec_t),
                get_cos(heading_vec, rel_vec),
                get_sin(heading_vec, rel_vec),
                dist_matrix,  # [N+M, N+M]
            ],
            dim=0,
        )  # [5, N+M, N+M]

        return rpe.float()

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Custom collate function to handle variable numbers of actors/lanes.
        """
        B = len(batch)
        batch = from_numpy(batch)  # all numpy to tensor

        new_batch = dict()
        # 1. Standard keys aggregation
        for key in batch[0].keys():
            new_batch[key] = [x[key] for x in batch]

        # 2. Flatten Actors (for GNN processing)
        b_agent_dict = self._actor_gather(B, new_batch)

        # 3. Flatten Lanes
        lane_feats, lane_masks, lane_ctrs, lane_vecs = self._lane_gather(B, new_batch)

        return {
            **b_agent_dict,
            "lane_ctrs": lane_ctrs,  # [B, Nmax, 2]  # lane_points
            "lane_vecs": lane_vecs,  # [B, Nmax, 2]
            "lane_feats": lane_feats,  # [B, Nmax, 10, 16]
            "lane_masks": lane_masks,  # [B, Nmax]
            "rpe": new_batch["rpe"],  # list of [5, N, N]
            "scenario_id": new_batch["scenario_id"],
            "city": new_batch["city"],
        }

    def _actor_gather(self, batch_size, batch):
        """Flattens actor features from a batch into a single tensor."""
        num_agents = [x.shape[0] for x in batch["agent_history"]]
        max_agents = max(num_agents)

        b_agent_history = torch.zeros(
            [batch_size, max_agents, self.history_steps - self.truncate_steps, 14]
        )  # truncate first 2 steps
        b_agent_history_mask = torch.zeros(
            [batch_size, max_agents, self.history_steps - self.truncate_steps],
            dtype=torch.bool,
        )
        b_agent_future_pos = torch.zeros([batch_size, max_agents, self.future_steps, 2])
        b_agent_future_ang = torch.zeros(
            [batch_size, max_agents, self.future_steps, 2]
        )  # TODO
        b_agent_future_mask = torch.zeros(
            [batch_size, max_agents, self.future_steps], dtype=torch.bool
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
            "focal_agent_point": torch.stack(batch["focal_agent_point"], dim=0),
            "focal_agent_rotation": torch.stack(batch["focal_agent_rotation"], dim=0),
            "agent_score_types": batch["agent_score_types"],
        }

    def _lane_gather(self, batch_size, graphs):
        """Flattens lane graph features from a batch."""
        lane_idcs = []
        lane_count = 0

        num_lanes = graphs["num_lanes"]
        max_lanes = max(num_lanes)

        lane_feats = torch.zeros([batch_size, max_lanes, 10, 16])
        lane_masks = torch.zeros([batch_size, max_lanes], dtype=torch.bool)

        lane_ctrs = torch.zeros([batch_size, max_lanes, 2])
        lane_vecs = torch.zeros([batch_size, max_lanes, 2])

        num_nodes_per_lane = graphs["node_ctrs"][0].shape[1]  # 10

        for i in range(batch_size):
            lane_idcs.append(
                torch.arange(lane_count, lane_count + graphs["num_lanes"][i])
            )
            lane_count += graphs["num_lanes"][i]

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
