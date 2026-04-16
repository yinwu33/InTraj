from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from datasets import (
    CANONICAL_AGENT_TYPES,
    CANONICAL_MAP_TYPES,
    MotionDataset,
    MotionLaneSegment,
    MotionPolylineFeature,
    MotionScenario,
    MotionTrack,
    StandardAgentConfig,
    StandardMapConfig,
    StandardizationConfig,
    build_standardized_record,
    get_standardized_cache_path,
    load_standardized_record,
    normalize_source_name,
    normalize_split_name,
    save_standardized_record,
)


_SIMPL_AGENT_TYPES = (
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "bus",
    "static",
    "unknown",
)
_SIMPL_AGENT_TYPE_TO_INDEX = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "motorcyclist": 3,
    "bus": 4,
    "static": 5,
    "unknown": 6,
}
_CANONICAL_TO_SIMPL_AGENT_TYPE = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "motorcyclist": "motorcyclist",
    "bus": "bus",
    "static": "static",
    "background": "unknown",
    "construction": "unknown",
    "riderless_bicycle": "unknown",
    "unknown": "unknown",
}
_YAW_LOSS_AGENT_TYPES = {"vehicle", "cyclist", "motorcyclist", "bus"}


@dataclass(frozen=True)
class _StandardizedMapFeatureRecord:
    feature_id: str
    feature_type: str
    points: np.ndarray
    is_intersection: bool = False


def _dataclass_kwargs(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    field_names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in field_names}


def _build_standardization_config(payload: dict[str, Any]) -> StandardizationConfig:
    payload = dict(payload)
    agents_payload = dict(payload.pop("agents", {}))
    map_payload = dict(payload.pop("map", {}))
    agents_cfg = StandardAgentConfig(
        **_dataclass_kwargs(StandardAgentConfig, agents_payload)
    )
    map_cfg = StandardMapConfig(**_dataclass_kwargs(StandardMapConfig, map_payload))
    return StandardizationConfig(
        agents=agents_cfg,
        map=map_cfg,
        **_dataclass_kwargs(StandardizationConfig, payload),
    )


def _build_motion_dataset(
    source: str,
    data_root: str,
    split: str,
    builder_kwargs: dict[str, Any],
) -> MotionDataset:
    normalized_source = str(source).lower()
    if normalized_source == "av2":
        return MotionDataset.create_from_av2(
            data_root=data_root,
            split=split,
            **builder_kwargs,
        )
    if normalized_source == "waymo":
        return MotionDataset.create_from_waymo(
            data_root=data_root,
            split=split,
            **builder_kwargs,
        )
    raise ValueError(f"Unsupported MotionDataset source: {source}")


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _rotation_from_heading(theta: np.ndarray) -> np.ndarray:
    cos = np.cos(theta)
    sin = np.sin(theta)
    rot = np.zeros(theta.shape + (2, 2), dtype=np.float32)
    rot[..., 0, 0] = cos
    rot[..., 0, 1] = -sin
    rot[..., 1, 0] = sin
    rot[..., 1, 1] = cos
    return rot


def _nearest_fill(values: np.ndarray, valid_mask: np.ndarray, fill_value: float) -> np.ndarray:
    filled = np.array(values, dtype=np.float32, copy=True)
    finite_mask = np.isfinite(filled).all(axis=-1) if filled.ndim > 1 else np.isfinite(filled)
    valid = np.asarray(valid_mask, dtype=bool) & finite_mask
    if not valid.any():
        filled[...] = fill_value
        return filled

    last_value = None
    for idx in range(filled.shape[0]):
        if valid[idx]:
            last_value = filled[idx].copy()
        elif last_value is not None:
            filled[idx] = last_value

    last_value = None
    for idx in range(filled.shape[0] - 1, -1, -1):
        if valid[idx]:
            last_value = filled[idx].copy()
        elif last_value is not None:
            filled[idx] = last_value

    if filled.ndim > 1:
        missing = ~np.isfinite(filled).all(axis=-1)
        filled[missing] = fill_value
    else:
        filled[~np.isfinite(filled)] = fill_value
    return filled


def _reference_index(valid_mask: np.ndarray, observed_mask: np.ndarray, current_index: int) -> int | None:
    if 0 <= current_index < observed_mask.shape[0] and observed_mask[current_index]:
        return int(current_index)
    observed_prefix = np.flatnonzero(observed_mask[: current_index + 1])
    if observed_prefix.size > 0:
        return int(observed_prefix[-1])
    valid_prefix = np.flatnonzero(valid_mask[: current_index + 1])
    if valid_prefix.size > 0:
        return int(valid_prefix[-1])
    all_valid = np.flatnonzero(valid_mask)
    if all_valid.size > 0:
        return int(all_valid[0])
    return None


def _safe_unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _score_label(is_ego: bool, is_target: bool, is_interest: bool) -> str:
    if is_ego:
        return "ego"
    if is_target:
        return "target"
    if is_interest:
        return "interest"
    return "context"


def _xy_to_xyz(xy_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    xyz = np.full((xy_values.shape[0], 3), np.nan, dtype=np.float32)
    finite_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(xy_values).all(axis=-1)
    xyz[finite_mask, :2] = xy_values[finite_mask].astype(np.float32)
    xyz[finite_mask, 2] = 0.0
    return xyz


def _rebuild_motion_scenario_from_record(record: dict[str, Any]) -> MotionScenario:
    agent_ids = list(record["agent_ids"])
    agent_positions = np.asarray(record["agent_positions"], dtype=np.float32)
    agent_velocities = np.asarray(record["agent_velocities"], dtype=np.float32)
    agent_headings = np.asarray(record["agent_headings"], dtype=np.float32)
    agent_valid_mask = np.asarray(record["agent_valid_mask"], dtype=bool)
    agent_observed_mask = np.asarray(record["agent_observed_mask"], dtype=bool)
    agent_types = np.asarray(record["agent_types"], dtype=np.int64)
    agent_is_ego = np.asarray(record["agent_is_ego"], dtype=bool)
    agent_is_target = np.asarray(record["agent_is_target"], dtype=bool)
    agent_is_interest = np.asarray(record["agent_is_interest"], dtype=bool)

    agent_size = record.get("agent_size")
    agent_size_valid_mask = record.get("agent_size_valid_mask")
    if agent_size is None:
        agent_size = np.zeros((len(agent_ids), 3), dtype=np.float32)
    else:
        agent_size = np.asarray(agent_size, dtype=np.float32)
    if agent_size_valid_mask is None:
        agent_size_valid_mask = np.zeros((len(agent_ids),), dtype=bool)
    else:
        agent_size_valid_mask = np.asarray(agent_size_valid_mask, dtype=bool)

    tracks: list[MotionTrack] = []
    for agent_idx, track_id in enumerate(agent_ids):
        sizes = np.full((agent_positions.shape[1], 3), np.nan, dtype=np.float32)
        if agent_idx < agent_size.shape[0] and agent_size_valid_mask[agent_idx]:
            sizes[:] = agent_size[agent_idx]

        tracks.append(
            MotionTrack(
                track_id=str(track_id),
                object_type=CANONICAL_AGENT_TYPES[int(agent_types[agent_idx])],
                category=None,
                positions=_xy_to_xyz(agent_positions[agent_idx], agent_valid_mask[agent_idx]),
                headings=agent_headings[agent_idx].astype(np.float32),
                velocities=_xy_to_xyz(
                    agent_velocities[agent_idx],
                    np.isfinite(agent_velocities[agent_idx]).all(axis=-1),
                ),
                sizes=sizes,
                valid_mask=agent_valid_mask[agent_idx],
                observed_mask=agent_observed_mask[agent_idx],
                is_ego=bool(agent_is_ego[agent_idx]),
                is_focal=False,
                is_prediction_target=bool(agent_is_target[agent_idx]),
                is_object_of_interest=bool(agent_is_interest[agent_idx]),
                metadata={"standardized": True},
            )
        )

    map_ids = list(record["map_ids"])
    map_types = np.asarray(record["map_types"], dtype=np.int64)
    map_points = np.asarray(record["map_points"], dtype=np.float32)
    map_valid_mask = np.asarray(record["map_valid_mask"], dtype=bool)
    map_is_intersection = np.asarray(record["map_is_intersection"], dtype=bool)

    lane_segments: list[MotionLaneSegment] = []
    road_lines: list[MotionPolylineFeature] = []
    road_edges: list[MotionPolylineFeature] = []
    map_feature_records: list[_StandardizedMapFeatureRecord] = []

    for feature_idx, feature_id in enumerate(map_ids):
        valid_points = map_points[feature_idx][map_valid_mask[feature_idx]].astype(np.float32)
        if valid_points.shape[0] < 2:
            continue

        feature_type = CANONICAL_MAP_TYPES[int(map_types[feature_idx])]
        is_intersection = bool(map_is_intersection[feature_idx])
        map_feature_records.append(
            _StandardizedMapFeatureRecord(
                feature_id=str(feature_id),
                feature_type=feature_type,
                points=valid_points,
                is_intersection=is_intersection,
            )
        )

        if feature_type == "lane_centerline":
            lane_segments.append(
                MotionLaneSegment(
                    lane_id=str(feature_id),
                    centerline=valid_points,
                    is_intersection=is_intersection,
                    metadata={"standardized": True},
                )
            )
        elif feature_type == "road_edge":
            road_edges.append(
                MotionPolylineFeature(
                    feature_id=str(feature_id),
                    feature_type=feature_type,
                    points=valid_points,
                    metadata={"standardized": True},
                )
            )
        else:
            road_lines.append(
                MotionPolylineFeature(
                    feature_id=str(feature_id),
                    feature_type=feature_type,
                    points=valid_points,
                    metadata={"standardized": True},
                )
            )

    standardization_metadata = dict(record["standardization_metadata"])
    standardization_metadata["map_feature_records"] = tuple(map_feature_records)

    metadata = dict(record.get("metadata", {}))
    metadata["standardization"] = standardization_metadata

    sdc_track_id = next(
        (track.track_id for track in tracks if track.is_ego),
        None,
    )

    return MotionScenario(
        scenario_id=str(record["scenario_id"]),
        source=normalize_source_name(record["source"]),
        split=normalize_split_name(record["split"]),
        timestamps_seconds=np.asarray(record["timestamps_seconds"], dtype=np.float32),
        current_time_index=int(record["current_time_index"]),
        tracks=tracks,
        lane_segments=lane_segments,
        road_lines=road_lines,
        road_edges=road_edges,
        city_name=record.get("city_name"),
        focal_track_id=record.get("primary_target_track_id"),
        sdc_track_id=sdc_track_id,
        metadata=metadata,
    )


class _SimplMotionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        source: str,
        split: str,
        cache_root: str,
        standardization_config: StandardizationConfig,
        builder_kwargs: dict[str, Any] | None = None,
        truncate_steps: int = 2,
        rpe_radius: float = 100.0,
    ) -> None:
        super().__init__()
        self.dataset = _build_motion_dataset(
            source=source,
            data_root=data_root,
            split=split,
            builder_kwargs=builder_kwargs or {},
        )
        self.source = normalize_source_name(source)
        self.split = normalize_split_name(split)
        self.cache_root = Path(cache_root).expanduser()
        self.history_steps = int(standardization_config.history_steps)
        self.future_steps = int(standardization_config.future_steps)
        self.map_feature_dim = len(CANONICAL_MAP_TYPES)
        self.agent_feature_dim = len(_SIMPL_AGENT_TYPES)
        self.truncate_steps = int(truncate_steps)
        self.rpe_radius = float(rpe_radius)
        self.standardization_config = standardization_config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scenario_ref = self.dataset.scenario_refs[idx]
        scenario_id = scenario_ref.scenario_id
        if scenario_id is None:
            raise ValueError(f"Scenario reference at index {idx} is missing scenario_id")

        cache_path = get_standardized_cache_path(
            self.cache_root,
            source=self.source,
            split=self.split,
            scenario_id=scenario_id,
        )
        if cache_path.exists():
            record = load_standardized_record(cache_path)
        else:
            raw_scenario = self.dataset[idx]
            record = build_standardized_record(
                raw_scenario,
                config=self.standardization_config,
                split=self.split,
            )
            save_standardized_record(cache_path, record)
        return self._record_to_sample(record)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}

        batch_size = len(batch)
        max_agents = max(item["agent_history"].shape[0] for item in batch)
        max_lanes = max(item["lane_feats"].shape[0] for item in batch)
        max_lane_nodes = max(item["lane_feats"].shape[1] for item in batch)
        history_steps = self.history_steps - self.truncate_steps

        b_agent_history = torch.zeros(
            (batch_size, max_agents, history_steps, 14), dtype=torch.float32
        )
        b_agent_history_mask = torch.zeros(
            (batch_size, max_agents, history_steps), dtype=torch.bool
        )
        b_agent_future_pos = torch.zeros(
            (batch_size, max_agents, self.future_steps, 2), dtype=torch.float32
        )
        b_agent_future_ang = torch.zeros(
            (batch_size, max_agents, self.future_steps, 2), dtype=torch.float32
        )
        b_agent_future_mask = torch.zeros(
            (batch_size, max_agents, self.future_steps), dtype=torch.bool
        )
        b_agent_last_pos = torch.zeros((batch_size, max_agents, 2), dtype=torch.float32)
        b_agent_last_rot = torch.zeros(
            (batch_size, max_agents, 2, 2), dtype=torch.float32
        )
        b_yaw_loss_mask = torch.zeros((batch_size, max_agents), dtype=torch.bool)
        b_target_mask = torch.zeros((batch_size, max_agents), dtype=torch.bool)
        b_agent_type = torch.full((batch_size, max_agents), -1, dtype=torch.long)

        b_lane_feats = torch.zeros(
            (batch_size, max_lanes, max_lane_nodes, 16), dtype=torch.float32
        )
        b_lane_masks = torch.zeros((batch_size, max_lanes), dtype=torch.bool)
        b_lane_ctrs = torch.zeros((batch_size, max_lanes, 2), dtype=torch.float32)
        b_lane_vecs = torch.zeros((batch_size, max_lanes, 2), dtype=torch.float32)

        scenario_ids: list[str] = []
        cities: list[str | None] = []
        agent_ids: list[list[str]] = []
        agent_score_types: list[list[str]] = []
        rpe_list: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        motion_samples: list[MotionScenario] = []

        for batch_idx, item in enumerate(batch):
            num_agents = item["agent_history"].shape[0]
            num_lanes = item["lane_feats"].shape[0]
            num_lane_nodes = item["lane_feats"].shape[1]

            b_agent_history[batch_idx, :num_agents] = item["agent_history"]
            b_agent_history_mask[batch_idx, :num_agents] = item["agent_history_mask"]
            b_agent_future_pos[batch_idx, :num_agents] = item["agent_future_pos"]
            b_agent_future_ang[batch_idx, :num_agents] = item["agent_future_ang"]
            b_agent_future_mask[batch_idx, :num_agents] = item["agent_future_mask"]
            b_agent_last_pos[batch_idx, :num_agents] = item["agent_last_pos"]
            b_agent_last_rot[batch_idx, :num_agents] = item["agent_last_rot"]
            b_yaw_loss_mask[batch_idx, :num_agents] = item["yaw_loss_mask"]
            b_target_mask[batch_idx, :num_agents] = item["target_mask"]
            b_agent_type[batch_idx, :num_agents] = item["agent_type"]

            b_lane_feats[batch_idx, :num_lanes, :num_lane_nodes] = item["lane_feats"]
            b_lane_masks[batch_idx, :num_lanes] = True
            b_lane_ctrs[batch_idx, :num_lanes] = item["lane_ctrs"]
            b_lane_vecs[batch_idx, :num_lanes] = item["lane_vecs"]

            scenario_ids.append(item["scenario_id"])
            cities.append(item["city"])
            agent_ids.append(item["agent_ids"])
            agent_score_types.append(item["agent_score_types"])
            rpe_list.append(item["rpe"])
            metadata.append(item["metadata"])
            motion_samples.append(item["motion_sample"])

        focal_agent_point = torch.stack([item["focal_agent_point"] for item in batch], dim=0)
        focal_agent_rotation = torch.stack(
            [item["focal_agent_rotation"] for item in batch], dim=0
        )

        return {
            "agent_history": b_agent_history,
            "agent_history_mask": b_agent_history_mask,
            "agent_future_pos": b_agent_future_pos,
            "agent_future_ang": b_agent_future_ang,
            "agent_future_mask": b_agent_future_mask,
            "agent_last_pos": b_agent_last_pos,
            "agent_last_rot": b_agent_last_rot,
            "yaw_loss_mask": b_yaw_loss_mask,
            "target_mask": b_target_mask,
            "agent_type": b_agent_type,
            "agent_ids": agent_ids,
            "agent_score_types": agent_score_types,
            "lane_feats": b_lane_feats,
            "lane_masks": b_lane_masks,
            "lane_ctrs": b_lane_ctrs,
            "lane_vecs": b_lane_vecs,
            "rpe": rpe_list,
            "focal_agent_point": focal_agent_point,
            "focal_agent_rotation": focal_agent_rotation,
            "scenario_id": scenario_ids,
            "city": cities,
            "metadata": metadata,
            "motion_samples": motion_samples,
        }

    def _record_to_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        positions = np.asarray(record["agent_positions"], dtype=np.float32)
        velocities = np.asarray(record["agent_velocities"], dtype=np.float32)
        headings = np.asarray(record["agent_headings"], dtype=np.float32)
        valid_mask = np.asarray(record["agent_valid_mask"], dtype=bool)
        observed_mask = np.asarray(record["agent_observed_mask"], dtype=bool)
        is_ego = np.asarray(record["agent_is_ego"], dtype=bool)
        is_target = np.asarray(record["agent_is_target"], dtype=bool)
        is_interest = np.asarray(record["agent_is_interest"], dtype=bool)
        agent_type_indices = np.asarray(record["agent_types"], dtype=np.int64)

        current_index = (
            int(record["current_time_index"])
            if record["current_time_index"] is not None
            else self.history_steps - 1
        )

        keep_indices: list[int] = []
        reference_indices: list[int] = []
        distances: list[float] = []

        for idx in range(positions.shape[0]):
            reference_index = _reference_index(valid_mask[idx], observed_mask[idx], current_index)
            if reference_index is None:
                continue
            filled_pos = _nearest_fill(positions[idx], valid_mask[idx], fill_value=0.0)
            keep_indices.append(idx)
            reference_indices.append(reference_index)
            distances.append(float(np.linalg.norm(filled_pos[reference_index])))

        if not keep_indices:
            raise ValueError(
                f"Scenario {scenario.scenario_id} has no valid agents after standardization"
            )

        order = sorted(
            range(len(keep_indices)),
            key=lambda item_idx: (
                0 if is_ego[keep_indices[item_idx]] else 1,
                0 if is_target[keep_indices[item_idx]] else 1,
                0 if is_interest[keep_indices[item_idx]] else 1,
                distances[item_idx],
                str(record["agent_ids"][keep_indices[item_idx]]),
            ),
        )
        keep_indices = [keep_indices[idx] for idx in order]
        reference_indices = [reference_indices[idx] for idx in order]

        positions = positions[keep_indices]
        velocities = velocities[keep_indices]
        headings = headings[keep_indices]
        valid_mask = valid_mask[keep_indices]
        is_ego = is_ego[keep_indices]
        is_target = is_target[keep_indices]
        is_interest = is_interest[keep_indices]
        agent_type_indices = agent_type_indices[keep_indices]
        agent_ids = [record["agent_ids"][idx] for idx in keep_indices]

        agent_count = len(keep_indices)

        positions_filled = np.stack(
            [_nearest_fill(positions[idx], valid_mask[idx], fill_value=0.0) for idx in range(agent_count)],
            axis=0,
        )
        headings_filled = np.stack(
            [_nearest_fill(headings[idx], valid_mask[idx], fill_value=0.0) for idx in range(agent_count)],
            axis=0,
        )
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)

        last_pos = np.stack(
            [positions_filled[idx, reference_indices[idx]] for idx in range(agent_count)],
            axis=0,
        ).astype(np.float32)
        last_heading = np.asarray(
            [headings_filled[idx, reference_indices[idx]] for idx in range(agent_count)],
            dtype=np.float32,
        )
        last_rot = _rotation_from_heading(last_heading)

        positions_local = np.matmul(positions_filled - last_pos[:, None, :], last_rot)
        headings_local = _wrap_angle(headings_filled - last_heading[:, None])
        velocities_local = np.matmul(velocities, last_rot)

        history_pos_local = positions_local[:, : self.history_steps]
        history_displacement = np.zeros_like(history_pos_local, dtype=np.float32)
        history_displacement[:, 1:] = history_pos_local[:, 1:] - history_pos_local[:, :-1]

        agent_type_onehot = np.zeros((agent_count, self.agent_feature_dim), dtype=np.float32)
        simpl_agent_type_indices = np.zeros((agent_count,), dtype=np.int64)
        yaw_loss_mask = np.zeros((agent_count,), dtype=bool)
        for idx, canonical_type_idx in enumerate(agent_type_indices):
            canonical_type = CANONICAL_AGENT_TYPES[int(canonical_type_idx)]
            simpl_type = _CANONICAL_TO_SIMPL_AGENT_TYPE.get(canonical_type, "unknown")
            simpl_idx = _SIMPL_AGENT_TYPE_TO_INDEX[simpl_type]
            simpl_agent_type_indices[idx] = simpl_idx
            agent_type_onehot[idx, simpl_idx] = 1.0
            yaw_loss_mask[idx] = simpl_type in _YAW_LOSS_AGENT_TYPES

        history_feat = np.concatenate(
            [
                history_displacement,
                np.cos(headings_local[:, : self.history_steps])[..., None],
                np.sin(headings_local[:, : self.history_steps])[..., None],
                velocities_local[:, : self.history_steps],
                np.repeat(agent_type_onehot[:, None, :], self.history_steps, axis=1),
                valid_mask[:, : self.history_steps, None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        future_pos = positions_filled[:, self.history_steps : self.history_steps + self.future_steps]
        future_heading = headings_filled[
            :, self.history_steps : self.history_steps + self.future_steps
        ]
        future_ang = np.stack(
            [np.cos(future_heading), np.sin(future_heading)],
            axis=-1,
        ).astype(np.float32)
        future_mask = valid_mask[
            :, self.history_steps : self.history_steps + self.future_steps
        ].astype(bool)

        lane_feats, lane_ctrs, lane_vecs = self._build_map_features(record)
        rpe = self._build_rpe(last_pos, last_heading, lane_ctrs, lane_vecs)

        return {
            "agent_history": torch.from_numpy(history_feat[:, self.truncate_steps :]),
            "agent_history_mask": torch.from_numpy(
                valid_mask[:, self.truncate_steps : self.history_steps]
            ).bool(),
            "agent_future_pos": torch.from_numpy(future_pos.astype(np.float32)),
            "agent_future_ang": torch.from_numpy(future_ang),
            "agent_future_mask": torch.from_numpy(future_mask),
            "agent_last_pos": torch.from_numpy(last_pos),
            "agent_last_rot": torch.from_numpy(last_rot),
            "yaw_loss_mask": torch.from_numpy(yaw_loss_mask),
            "target_mask": torch.from_numpy(is_target.astype(bool)),
            "agent_type": torch.from_numpy(simpl_agent_type_indices),
            "agent_ids": agent_ids,
            "agent_score_types": [
                _score_label(bool(agent_is_ego), bool(agent_is_target), bool(agent_is_interest))
                for agent_is_ego, agent_is_target, agent_is_interest in zip(
                    is_ego, is_target, is_interest
                )
            ],
            "lane_feats": torch.from_numpy(lane_feats),
            "lane_ctrs": torch.from_numpy(lane_ctrs),
            "lane_vecs": torch.from_numpy(lane_vecs),
            "rpe": torch.from_numpy(rpe),
            "focal_agent_point": torch.zeros(2, dtype=torch.float32),
            "focal_agent_rotation": torch.eye(2, dtype=torch.float32),
            "scenario_id": str(record["scenario_id"]),
            "city": record.get("city_name"),
            "metadata": dict(record.get("metadata", {})),
            "motion_sample": _rebuild_motion_scenario_from_record(record),
        }

    def _build_map_features(
        self,
        record: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_points = np.asarray(record["map_points"], dtype=np.float32)
        map_valid_mask = np.asarray(record["map_valid_mask"], dtype=bool)
        map_types = np.asarray(record["map_types"], dtype=np.int64)
        map_is_intersection = np.asarray(record["map_is_intersection"], dtype=bool)

        keep_indices = [idx for idx in range(map_points.shape[0]) if map_valid_mask[idx].sum() >= 2]
        if not keep_indices:
            return (
                np.zeros((0, map_points.shape[1] - 1, 16), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        map_points = map_points[keep_indices]
        map_valid_mask = map_valid_mask[keep_indices]
        map_types = map_types[keep_indices]
        map_is_intersection = map_is_intersection[keep_indices]

        polyline_count = map_points.shape[0]
        num_nodes = map_points.shape[1] - 1

        points_filled = map_points.copy()
        lane_ctrs = np.zeros((polyline_count, 2), dtype=np.float32)
        lane_vecs = np.zeros((polyline_count, 2), dtype=np.float32)

        for idx in range(polyline_count):
            valid_count = int(map_valid_mask[idx].sum())
            valid_points = map_points[idx, :valid_count]
            lane_ctrs[idx] = valid_points.mean(axis=0)
            lane_vecs[idx] = _safe_unit_vector(valid_points[-1] - valid_points[0])
            if valid_count < map_points.shape[1]:
                points_filled[idx, valid_count:] = valid_points[-1]

        lane_rot = _rotation_from_heading(np.arctan2(lane_vecs[:, 1], lane_vecs[:, 0]))
        points_local = np.matmul(points_filled - lane_ctrs[:, None, :], lane_rot)

        node_ctrs = (points_local[:, :-1] + points_local[:, 1:]) / 2.0
        node_vecs = points_local[:, 1:] - points_local[:, :-1]

        map_type_onehot = np.zeros(
            (polyline_count, len(CANONICAL_MAP_TYPES)),
            dtype=np.float32,
        )
        map_type_onehot[np.arange(polyline_count), map_types] = 1.0

        lane_feats = np.concatenate(
            [
                node_ctrs.astype(np.float32),
                node_vecs.astype(np.float32),
                map_is_intersection[:, None, None].astype(np.float32).repeat(num_nodes, axis=1),
                map_type_onehot[:, None, :].repeat(num_nodes, axis=1),
                np.zeros((polyline_count, num_nodes, 4), dtype=np.float32),
            ],
            axis=-1,
        )
        return lane_feats.astype(np.float32), lane_ctrs, lane_vecs

    def _build_rpe(
        self,
        agent_last_pos: np.ndarray,
        agent_last_heading: np.ndarray,
        lane_ctrs: np.ndarray,
        lane_vecs: np.ndarray,
    ) -> np.ndarray:
        agent_heading_vec = np.stack(
            [np.cos(agent_last_heading), np.sin(agent_last_heading)],
            axis=-1,
        ).astype(np.float32)

        scene_points = np.concatenate([agent_last_pos, lane_ctrs], axis=0).astype(np.float32)
        scene_vectors = np.concatenate([agent_heading_vec, lane_vecs], axis=0).astype(np.float32)

        if scene_points.shape[0] == 0:
            return np.zeros((5, 0, 0), dtype=np.float32)

        diff = scene_points[None, :, :] - scene_points[:, None, :]
        dist_matrix = np.linalg.norm(diff, axis=-1).astype(np.float32)
        dist_matrix = dist_matrix / max(self.rpe_radius, 1e-6) * 2.0

        vectors_a = scene_vectors[None, :, :]
        vectors_b = scene_vectors[:, None, :]

        def _cos(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-10
            return (v1 * v2).sum(axis=-1) / denom

        def _sin(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-10
            return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / denom

        return np.stack(
            [
                _cos(vectors_a, vectors_b),
                _sin(vectors_a, vectors_b),
                _cos(vectors_a, diff),
                _sin(vectors_a, diff),
                dist_matrix,
            ],
            axis=0,
        ).astype(np.float32)


class SimplDatamodule(pl.LightningDataModule):
    """Unified MotionDataset-backed datamodule for the current Simpl batch contract."""

    def __init__(
        self,
        data_root: str,
        dataset: OmegaConf,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        dataset_cfg = OmegaConf.to_container(dataset, resolve=True)
        if not isinstance(dataset_cfg, dict):
            raise TypeError(f"Expected datamodule.dataset to resolve to a dict, got {type(dataset_cfg)!r}")

        self.data_root = data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.source = str(dataset_cfg.get("source", "av2"))
        self.builder_kwargs = dict(dataset_cfg.get("builder_kwargs", {}))
        self.cache_root = to_absolute_path(
            str(dataset_cfg.get("cache_root", "cache/motiondataset"))
        )
        standardization_payload = dict(dataset_cfg.get("standardization", {}))
        if "history_steps" not in standardization_payload and "history_steps" in dataset_cfg:
            standardization_payload["history_steps"] = dataset_cfg["history_steps"]
        if "future_steps" not in standardization_payload and "future_steps" in dataset_cfg:
            standardization_payload["future_steps"] = dataset_cfg["future_steps"]
        self.standardization_config = _build_standardization_config(standardization_payload)
        self.truncate_steps = int(dataset_cfg.get("truncate_steps", 2))
        self.rpe_radius = float(
            dataset_cfg.get(
                "radius",
                self.standardization_config.map.range_m,
            )
        )

        self.train_dataset: _SimplMotionDataset | None = None
        self.val_dataset: _SimplMotionDataset | None = None
        self.test_dataset: _SimplMotionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(self.train_split)
            self.val_dataset = self._build_dataset(self.val_split)
        if stage in ("test", None):
            self.test_dataset = self._build_dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def _build_dataset(self, split: str) -> _SimplMotionDataset:
        return _SimplMotionDataset(
            data_root=to_absolute_path(str(self.data_root)),
            source=self.source,
            split=split,
            cache_root=self.cache_root,
            standardization_config=self.standardization_config,
            builder_kwargs=self.builder_kwargs,
            truncate_steps=self.truncate_steps,
            rpe_radius=self.rpe_radius,
        )
