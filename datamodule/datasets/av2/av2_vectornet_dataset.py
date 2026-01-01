from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .av2_base_dataset import AV2BaseDataset


_SCORE_TYPE_ID_TO_NAME = {
    0: "fragment",
    1: "unscore",
    2: "score",
    3: "focal",
    4: "av",
}


class AV2VectorNetDataset(AV2BaseDataset):
    """Dataset that builds VectorNet-friendly tensors from AV2 raw scenarios."""

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
        lane_points: int = 20,
        lane_agent_k: int = 3,
        lane_radius: float = 150.0,
        agent_radius: float = 30.0,
        preprocess: bool = False,
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


        self.lane_points = lane_points
        self.lane_agent_k = lane_agent_k
        self.lane_radius = lane_radius
        self.agent_radius = agent_radius

        # folder under data_root / split
        self.log_dirs = sorted((self.data_root / split).glob("*"))
        self.cache_dir = (
            Path(preprocess_dir) / "av2_vectornet" / split
            if preprocess_dir is not None
            else None
        )
        self.preprocess = preprocess
        if self.preprocess:
            if self.cache_dir is None:
                raise ValueError("preprocess_dir must be provided when preprocess=True")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index) -> dict:
        log_id = self.log_dirs[index].name
        cache_file = (
            self.cache_dir / f"{log_id}.pt"
            if (self.preprocess and self.cache_dir is not None)
            else None
        )

        # * load from cache if exists
        if cache_file is not None and cache_file.exists():
            try:
                sample = torch.load(cache_file, map_location="cpu", weights_only=False)
                return sample
            except Exception:
                print(f"Warning: failed to load cache file {cache_file}, rebuilding...")

        data = self.get_scene_centric_data(index, centric="focal")
        sample = self._build_sample_from_scene(data)
        if cache_file is not None:
            torch.save(sample, cache_file)
        return sample

    def _build_sample_from_scene(self, data: Dict[str, Any]) -> Dict[str, Any]:
        hist_pos = data["agent_positions"][:, : self.hist_steps, :]
        hist_vel = data["agent_velocities"][:, : self.hist_steps, :]
        hist_ang = data["agent_heading_angles"][:, : self.hist_steps]
        hist_valid_mask = data["agent_valid_masks"][:, : self.hist_steps]

        hist_feats = np.concatenate(
            [
                hist_pos,
                hist_vel,
                np.sin(hist_ang)[..., None],
                np.cos(hist_ang)[..., None],
                hist_valid_mask[..., None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        fut_pos = data["agent_positions"][
            :, self.hist_steps : self.hist_steps + self.fut_steps, :
        ]
        fut_valid_mask = data["agent_valid_masks"][
            :, self.hist_steps : self.hist_steps + self.fut_steps
        ]

        agent_history_tensor = torch.from_numpy(hist_feats)
        agent_history_masks_tensor = torch.from_numpy(hist_valid_mask)
        agent_future_tensor = torch.from_numpy(fut_pos)
        agent_future_masks_tensor = torch.from_numpy(fut_valid_mask)
        agent_last_pos_tensor = torch.from_numpy(data["agent_last_positions"])

        lane_points_tensor = self._prepare_lane_points(data["lane_points"])

        agent_edge_index = self._build_agent_agent_edges(agent_last_pos_tensor)
        lane_edge_index = self._build_lane_lane_edges(lane_points_tensor)
        edge_index_lane_agent = self._build_lane_agent_edges(
            lane_points_tensor, agent_last_pos_tensor
        )

        sample = {
            "scenario_id": data["scenario_id"],
            "lane_points": lane_points_tensor.float(),
            "agent_history": agent_history_tensor.float(),
            "agent_history_mask": agent_history_masks_tensor.bool(),
            "agent_future": agent_future_tensor.float(),
            "agent_future_mask": agent_future_masks_tensor.bool(),
            "agent_last_pos": agent_last_pos_tensor.float(),
            "target_agent_idx": torch.tensor(0, dtype=torch.long),
            "target_gt": agent_future_tensor[0].float(),
            "target_last_pos": agent_last_pos_tensor[0].float(),
            "edge_index_agent_to_agent": agent_edge_index.long(),
            "edge_index_lane_to_lane": lane_edge_index.long(),
            "edge_index_lane_to_agent": edge_index_lane_agent.long(),
            "agent_types": data["agent_types"],
            "agent_score_types": data["agent_score_types"],
        }
        return sample

    def _prepare_lane_points(self, lane_points: np.ndarray) -> torch.Tensor:
        """lane points are in focal centric frame.
        sorting, and filter out by lane_radius and max_lanes
        """
        if lane_points is None or lane_points.shape[0] == 0:
            return torch.zeros((0, self.lane_points, 2), dtype=torch.float32)

        lane_centers = lane_points.mean(axis=1)  # [num_lanes, 2]
        dists = np.linalg.norm(lane_centers, axis=1)  # [num_lanes]
        order = np.argsort(dists)

        selected: List[np.ndarray] = []
        for idx in order:
            if len(selected) >= self.max_lanes:
                break
            if dists[idx] > self.lane_radius:
                continue
            selected.append(lane_points[idx])

        if len(selected) == 0:
            closest_idx = int(order[0])
            selected.append(lane_points[closest_idx])

        lane_tensor = torch.from_numpy(np.stack(selected, axis=0)).float()
        return lane_tensor

    def _build_agent_agent_edges(
        self, agent_last_positions: torch.Tensor
    ) -> torch.Tensor:
        # agent_last_positions: [num_agents, 2]

        if agent_last_positions.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)

        last_pos_arr = agent_last_positions.detach().cpu().numpy()
        edges: List[Tuple[int, int]] = []
        num_agents = last_pos_arr.shape[0]
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                dist = np.linalg.norm(last_pos_arr[i] - last_pos_arr[j])
                if dist <= self.agent_radius:
                    edges.append((i, j))

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _build_lane_lane_edges(self, lane_points: torch.Tensor) -> torch.Tensor:
        # lane_points: [num_lanes, lane_points, 2]
        num_lanes = lane_points.shape[0]
        if num_lanes == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edges: List[Tuple[int, int]] = []

        lane_centers = lane_points[:, :, :2].mean(dim=1).detach().cpu().numpy()
        for i in range(num_lanes):
            for j in range(num_lanes):
                if i == j:
                    continue
                dist = np.linalg.norm(lane_centers[i] - lane_centers[j])
                if dist <= self.lane_radius:
                    edges.append((i, j))

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _build_lane_agent_edges(
        self, lane_points: torch.Tensor, agent_last_positions: torch.Tensor
    ) -> torch.Tensor:
        lane_agent_edges: List[Tuple[int, int]] = []
        if lane_points.shape[0] > 0 and agent_last_positions.shape[0] > 0:
            lane_ref = lane_points[:, :, :2].mean(dim=1).detach().cpu().numpy()
            agent_arr = agent_last_positions.detach().cpu().numpy()
            dist_matrix = np.linalg.norm(
                agent_arr[:, None, :] - lane_ref[None, :, :], axis=-1
            )
            for agent_idx in range(dist_matrix.shape[0]):
                lane_order = np.argsort(dist_matrix[agent_idx])
                for lane_idx in lane_order[
                    : min(self.lane_agent_k, lane_points.shape[0])
                ]:
                    lane_agent_edges.append((lane_idx, agent_idx))
        edge_index_lane_agent = (
            torch.tensor(lane_agent_edges, dtype=torch.long).t().contiguous()
            if len(lane_agent_edges) > 0
            else torch.empty((2, 0), dtype=torch.long)
        )
        return edge_index_lane_agent

    def collate_fn(self, batch: List[Dict]):
        lane_offset = 0
        agent_offset = 0

        lane_points = []
        agent_history = []
        agent_history_mask = []
        agent_future = []
        agent_future_mask = []
        lane_lane_edges = []
        agent_agent_edges = []
        lane_agent_edges = []
        target_indices = []
        target_last_pos = []
        agent_last_pos = []
        target_gt = []
        scenario_ids = []
        centroid = []
        lane_counts = []
        agent_counts = []
        agent_types = []
        agent_score_types = []

        for sample in batch:
            lane_points.append(sample["lane_points"])
            agent_history.append(sample["agent_history"])
            agent_history_mask.append(sample["agent_history_mask"])
            agent_future.append(sample["agent_future"])
            agent_future_mask.append(sample["agent_future_mask"])

            lane_counts.append(int(sample["lane_points"].shape[0]))
            agent_counts.append(int(sample["agent_history"].shape[0]))

            if sample["edge_index_lane_to_lane"].numel() > 0:
                lane_lane_edges.append(sample["edge_index_lane_to_lane"] + lane_offset)
            if sample["edge_index_agent_to_agent"].numel() > 0:
                agent_agent_edges.append(
                    sample["edge_index_agent_to_agent"] + agent_offset
                )
            if sample["edge_index_lane_to_agent"].numel() > 0:
                adjusted = sample["edge_index_lane_to_agent"].clone()
                adjusted[0, :] += lane_offset
                adjusted[1, :] += agent_offset
                lane_agent_edges.append(adjusted)

            target_indices.append(sample["target_agent_idx"] + agent_offset)
            target_last_pos.append(sample["target_last_pos"])
            agent_last_pos.append(sample["agent_last_pos"])
            target_gt.append(sample["target_gt"])
            scenario_ids.append(sample.get("scenario_id", ""))

            agent_types.extend(sample["agent_types"])
            agent_score_types.extend(sample["agent_score_types"])

            lane_offset += sample["lane_points"].shape[0]
            agent_offset += sample["agent_history"].shape[0]

        def _concat_tensors(
            tensors: List[torch.Tensor], dim: int = 0, empty_shape=(0,)
        ) -> torch.Tensor:
            if len(tensors) == 0:
                return torch.zeros(
                    empty_shape, dtype=torch.float32 if dim == 0 else torch.long
                )
            return torch.cat(tensors, dim=dim)

        batch_dict = {
            "lane_points": _concat_tensors(
                lane_points, dim=0, empty_shape=(0, self.lane_points, 2)
            ).float(),
            "agent_history": _concat_tensors(
                agent_history, dim=0, empty_shape=(0, self.hist_steps, 7)
            ).float(),
            "agent_history_mask": _concat_tensors(
                agent_history_mask, dim=0, empty_shape=(0, self.hist_steps)
            ),
            "agent_future": _concat_tensors(
                agent_future, dim=0, empty_shape=(0, self.fut_steps, 2)
            ).float(),
            "agent_future_mask": _concat_tensors(
                agent_future_mask, dim=0, empty_shape=(0, self.fut_steps)
            ),
            "edge_index_lane_to_lane": _concat_tensors(
                lane_lane_edges, dim=1, empty_shape=(2, 0)
            ).long(),
            "edge_index_agent_to_agent": _concat_tensors(
                agent_agent_edges, dim=1, empty_shape=(2, 0)
            ).long(),
            "edge_index_lane_to_agent": _concat_tensors(
                lane_agent_edges, dim=1, empty_shape=(2, 0)
            ).long(),
            "target_agent_global_idx": (
                torch.stack(target_indices)
                if len(target_indices) > 0
                else torch.zeros(0, dtype=torch.long)
            ),
            "target_last_pos": (
                torch.stack(target_last_pos)
                if len(target_last_pos) > 0
                else torch.zeros((0, 2))
            ),
            "agent_last_pos": _concat_tensors(
                agent_last_pos, dim=0, empty_shape=(0, 2)
            ),
            "target_gt": (
                torch.stack(target_gt)
                if len(target_gt) > 0
                else torch.zeros((0, self.fut_steps, 2))
            ),
            "centroid": (
                torch.stack(centroid) if len(centroid) > 0 else torch.zeros((0, 2))
            ),
            "scenario_ids": scenario_ids,
            "lane_counts": lane_counts,
            "agent_counts": agent_counts,
            "agent_types": torch.tensor(agent_types, dtype=torch.long),
            "agent_score_types": torch.tensor(agent_score_types, dtype=torch.long),
        }
        return batch_dict


if __name__ == "__main__":
    ds = AV2VectorNetDataset(
        data_root="./data",
        split="mini_train",
        history_steps=50,
        future_steps=60,
    )
    print(f"len: {len(ds)}")
    _ = ds[0]
