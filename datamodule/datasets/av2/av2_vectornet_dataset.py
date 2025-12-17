from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap


@dataclass
class AgentSequence:
    features: np.ndarray  # [T, F]
    mask: np.ndarray  # [T] bool


class AV2VectorNetDataset(Dataset):
    """Dataset that builds VectorNet-friendly tensors from AV2 raw scenarios."""

    def __init__(
        self,
        data_root: str,
        split: str = "mini_train",
        preprocess: bool = False,
        preprocess_dir: str = None,
        history_steps: int = 50,
        future_steps: int = 60,
        max_agents: int = 64,
        max_lanes: int = 128,
        lane_points: int = 20,
        lane_agent_k: int = 3,
        lane_radius: float = 150.0,
        agent_radius: float = 30.0,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split

        self.preprocess = preprocess
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.lane_agent_k = lane_agent_k
        self.lane_radius = lane_radius
        self.agent_radius = agent_radius

        # folder under data_root / split
        self.log_dirs = sorted((self.data_root / split).glob("*"))
        self.cache_dir = (
            self.data_root if preprocess_dir is None else Path(preprocess_dir)
        )
        self.cache_dir = self.cache_dir / "cache" / split
        if self.preprocess:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.log_dirs)

    # --- helpers ------------------------------------------------------------------ #
    def _resample_polyline(self, polyline: np.ndarray) -> np.ndarray:
        """Uniformly sample a polyline to a fixed number of points."""
        if polyline.shape[0] == 0:
            return np.zeros((self.lane_points, 2), dtype=np.float32)
        if polyline.shape[0] == 1:
            return np.repeat(polyline[:, :2], self.lane_points, axis=0)

        dists = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
        cumulative = np.insert(np.cumsum(dists), 0, 0.0)
        target = np.linspace(0, cumulative[-1], self.lane_points)

        resampled = []
        for t in target:
            idx = np.searchsorted(cumulative, t)
            if idx == 0:
                resampled.append(polyline[0, :2])
                continue
            if idx >= len(cumulative):
                resampled.append(polyline[-1, :2])
                continue
            prev, nxt = cumulative[idx - 1], cumulative[idx]
            ratio = 0.0 if nxt == prev else (t - prev) / (nxt - prev)
            point = polyline[idx - 1, :2] + ratio * (
                polyline[idx, :2] - polyline[idx - 1, :2]
            )
            resampled.append(point)
        return np.stack(resampled, axis=0).astype(np.float32)

    def _agent_to_sequence(self, track, total_steps: int) -> AgentSequence:
        """Convert a Track to a dense sequence with forward filled positions."""
        feat = np.zeros((total_steps, 7), dtype=np.float32)
        mask = np.zeros(total_steps, dtype=bool)

        for state in track.object_states:
            if state.timestep >= total_steps:
                continue
            feat[state.timestep, 0:2] = state.position
            feat[state.timestep, 2:4] = state.velocity
            feat[state.timestep, 4] = np.sin(state.heading)
            feat[state.timestep, 5] = np.cos(state.heading)
            feat[state.timestep, 6] = 1.0 if state.observed else 0.0
            mask[state.timestep] = True

        # forward-fill to avoid empty coordinates
        for t in range(1, total_steps):
            if not mask[t]:
                feat[t, :6] = feat[t - 1, :6]
        return AgentSequence(features=feat, mask=mask)

    def _select_last_valid(
        self, positions: np.ndarray, mask: np.ndarray, horizon: int
    ) -> np.ndarray:
        valid = np.where(mask[:horizon])[0]
        idx = valid[-1] if len(valid) > 0 else max(horizon - 1, 0)
        return positions[idx]

    def _build_lane_graph(
        self, static_map: ArgoverseStaticMap, center_point: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lane_ids = static_map.get_scenario_lane_segment_ids()
        lane_geoms: List[np.ndarray] = []
        lane_dists: List[float] = []

        for lane_id in lane_ids:
            centerline = static_map.get_lane_segment_centerline(lane_id)
            lane_geoms.append(centerline)
            lane_dists.append(
                float(
                    np.min(
                        np.linalg.norm(
                            centerline[:, :2] - center_point[None, :], axis=1
                        )
                    )
                )
            )

        order = np.argsort(np.asarray(lane_dists))
        chosen_idx = order[: self.max_lanes]
        chosen_lane_ids = [lane_ids[i] for i in chosen_idx]
        lane_points = [
            self._resample_polyline(lane_geoms[i])
            for i in chosen_idx
            if lane_dists[i] <= self.lane_radius
        ]
        chosen_lane_ids = [
            lane_ids[i] for i in chosen_idx if lane_dists[i] <= self.lane_radius
        ]

        if len(lane_points) == 0 and len(lane_geoms) > 0:
            closest_idx = int(order[0])
            lane_points = [self._resample_polyline(lane_geoms[closest_idx])]
            chosen_lane_ids = [lane_ids[closest_idx]]

        id_to_local: Dict[int, int] = {
            lane_id: i for i, lane_id in enumerate(chosen_lane_ids)
        }
        lane_edges: List[Tuple[int, int]] = []
        for lane_id in chosen_lane_ids:
            src = id_to_local[lane_id]
            for succ in static_map.get_lane_segment_successor_ids(lane_id) or []:
                if succ in id_to_local:
                    lane_edges.append((src, id_to_local[succ]))
            for nbr in [
                static_map.get_lane_segment_left_neighbor_id(lane_id),
                static_map.get_lane_segment_right_neighbor_id(lane_id),
            ]:
                if nbr is not None and nbr in id_to_local:
                    lane_edges.append((src, id_to_local[nbr]))

        lane_points_tensor = (
            torch.from_numpy(np.stack(lane_points, axis=0)).float()
            if lane_points
            else torch.zeros((0, self.lane_points, 2), dtype=torch.float32)
        )
        if len(lane_edges) == 0:
            lane_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            lane_edge_index = (
                torch.tensor(lane_edges, dtype=torch.long).t().contiguous()
            )
        return lane_points_tensor, lane_edge_index

    # --- main --------------------------------------------------------------------- #
    def __getitem__(self, index) -> dict:
        log_dir = self.log_dirs[index]
        log_id = log_dir.name
        json_file = log_dir / f"log_map_archive_{log_id}.json"
        parquet_file = log_dir / f"scenario_{log_id}.parquet"
        cache_file = self.cache_dir / \
            f"{log_id}.pt" if self.preprocess else None

        if cache_file is not None and cache_file.exists():
            try:
                sample = torch.load(
                    cache_file, map_location="cpu", weights_only=False)
                return sample
            except Exception:
                print(
                    f"Warning: failed to load cache file {cache_file}, rebuilding...")

        if not json_file.exists():
            raise FileNotFoundError(f"{json_file} not found")
        if not parquet_file.exists():
            raise FileNotFoundError(f"{parquet_file} not found")

        static_map = ArgoverseStaticMap.from_json(json_file)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(
            parquet_file)

        total_steps = self.history_steps + self.future_steps

        target_track = None
        for track in scenario.tracks:
            if track.track_id == scenario.focal_track_id:
                target_track = track
                break
        if target_track is None:
            target_track = scenario.tracks[0]

        target_seq = self._agent_to_sequence(target_track, total_steps)
        target_history = target_seq.features[: self.history_steps]
        target_last_pos = self._select_last_valid(
            target_history[:, :2], target_seq.mask, self.history_steps
        )
        target_future = target_seq.features[
            self.history_steps: self.history_steps + self.future_steps, :2
        ]

        # gather other agents sorted by amount of history available
        agent_entries: List[Tuple[str, AgentSequence]] = []
        for track in scenario.tracks:
            seq = self._agent_to_sequence(track, total_steps)
            history_obs = int(seq.mask[: self.history_steps].sum())
            if history_obs == 0:
                continue
            agent_entries.append((track.track_id, seq))
        agent_entries.sort(
            key=lambda x: int(x[1].mask[: self.history_steps].sum()), reverse=True
        )

        agent_histories: List[torch.Tensor] = []
        agent_history_masks: List[torch.Tensor] = []
        agent_futures: List[torch.Tensor] = []
        agent_future_masks: List[torch.Tensor] = []
        agent_last_positions: List[np.ndarray] = []
        target_agent_idx = 0

        # * make sure target is at the first position
        ordered_agents: List[Tuple[str, AgentSequence]] = []
        for track_id, seq in agent_entries:
            if track_id == target_track.track_id:
                ordered_agents.insert(0, (track_id, seq))
            else:
                ordered_agents.append((track_id, seq))

        for idx_entry, (track_id, seq) in enumerate(ordered_agents):
            if len(agent_histories) >= self.max_agents:
                break
            agent_histories.append(torch.from_numpy(
                seq.features[: self.history_steps]))
            agent_history_masks.append(
                torch.from_numpy(
                    seq.mask[:self.history_steps])
            )

            agent_futures.append(
                torch.from_numpy(
                    seq.features[self.history_steps:self.history_steps+self.future_steps, :2])
            )
            agent_future_masks.append(
                torch.from_numpy(
                    seq.mask[self.history_steps:self.history_steps +
                             self.future_steps]
                )
            )

            agent_last_positions.append(
                self._select_last_valid(
                    seq.features[:, :2], seq.mask, self.history_steps
                )
            )
            if track_id == target_track.track_id:
                target_agent_idx = idx_entry

        agent_history_tensor = (
            torch.stack(agent_histories, dim=0)
            if agent_histories
            else torch.zeros((0, self.history_steps, 7))
        )
        agent_history_masks_tensor = (
            torch.stack(agent_history_masks, dim=0)
            if agent_history_masks
            else torch.zeros((0, self.history_steps))
        )
        agent_future_tensor = (
            torch.stack(agent_futures, dim=0)
            if agent_futures
            else torch.zeros((0, self.future_steps, 2))
        )
        agent_future_masks_tensor = (
            torch.stack(agent_future_masks, dim=0)
            if agent_future_masks
            else torch.zeros((0, self.future_steps))
        )
        
        agent_last_pos_tensor = (
            torch.from_numpy(np.stack(agent_last_positions, axis=0)).float()
            if len(agent_last_positions) > 0
            else torch.zeros((0, 2), dtype=torch.float)
        )

        lane_points, lane_edge_index = self._build_lane_graph(
            static_map, target_last_pos
        )

        # agent-agent fully connect nearby actors
        if len(agent_last_positions) == 0:
            agent_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            last_pos_arr = np.stack(agent_last_positions, axis=0)
            edges: List[Tuple[int, int]] = []
            for i in range(len(agent_last_positions)):
                for j in range(len(agent_last_positions)):
                    if i == j:
                        continue
                    dist = np.linalg.norm(last_pos_arr[i] - last_pos_arr[j])
                    if dist <= self.agent_radius:
                        edges.append((i, j))
            agent_edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
                if len(edges) > 0
                else torch.empty((2, 0), dtype=torch.long)
            )

        # lane-agent edges: connect each agent to k nearest lanes
        lane_agent_edges: List[Tuple[int, int]] = []
        if lane_points.shape[0] > 0 and len(agent_last_positions) > 0:
            lane_ref = lane_points[:, :, :2].mean(dim=1).numpy()
            agent_arr = np.stack(agent_last_positions, axis=0)
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

        sample = {
            "lane_points": lane_points.float(),  # [num_lanes, lane_points, 2]
            # [num_agents, history_steps, 7], feat: x, y, vx, vy, sin_heading, cos_heading, observed
            "agent_history": agent_history_tensor.float(),
            # [num_agents, history_steps]
            "agent_history_mask": agent_history_masks_tensor.bool(),
            # [num_agents, future_steps, 2]
            "agent_future": agent_future_tensor.float(),
            # [num_agents, future_steps]
            "agent_future_mask": agent_future_masks_tensor.bool(),
            # [2, num_lane_edges]
            "edge_index_lane_to_lane": lane_edge_index.long(),
            # [2, num_agent_edges]
            "edge_index_agent_to_agent": agent_edge_index.long(),
            # [2, num_lane_agent_edges]
            "edge_index_lane_to_agent": edge_index_lane_agent.long(),
            # int
            "target_agent_idx": torch.tensor(target_agent_idx, dtype=torch.long),
            # [2]
            "target_last_pos": torch.from_numpy(target_last_pos).float(),
            "agent_last_pos": agent_last_pos_tensor,  # [num_agents, 2]
            # [future_steps, 2]
            "target_gt": torch.from_numpy(target_future).float(),
            "scenario_id": scenario.scenario_id,  # str
            # [2] scene-centric
            "centroid": torch.from_numpy(target_last_pos).float(),
        }
        if cache_file is not None:
            torch.save(sample, cache_file)
        return sample
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

        for sample in batch:
            lane_points.append(sample["lane_points"])
            agent_history.append(sample["agent_history"])
            agent_history_mask.append(sample["agent_history_mask"])
            agent_future.append(sample["agent_future"])
            agent_future_mask.append(sample["agent_future_mask"])

            lane_counts.append(int(sample["lane_points"].shape[0]))
            agent_counts.append(int(sample["agent_history"].shape[0]))

            if sample["edge_index_lane_to_lane"].numel() > 0:
                lane_lane_edges.append(
                    sample["edge_index_lane_to_lane"] + lane_offset)
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
            centroid.append(sample["centroid"])

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
                agent_history, dim=0, empty_shape=(0, self.history_steps, 7)
            ).float(),
            "agent_history_mask": _concat_tensors(
                agent_history_mask, dim=0, empty_shape=(0, self.history_steps)
            ),
            "agent_future": _concat_tensors(
                agent_future, dim=0, empty_shape=(0, self.future_steps, 2)
            ).float(),
            "agent_future_mask": _concat_tensors(
                agent_future_mask, dim=0, empty_shape=(0, self.future_steps)
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
                agent_last_pos,
                dim=0,
                empty_shape=(0, 2)
            ),
            "target_gt": (
                torch.stack(target_gt)
                if len(target_gt) > 0
                else torch.zeros((0, self.future_steps, 2))
            ),
            "centroid": (
                torch.stack(centroid)
                if len(centroid) > 0
                else torch.zeros((0, 2))
            ),
            "scenario_ids": scenario_ids,
            "lane_counts": lane_counts,
            "agent_counts": agent_counts,
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
