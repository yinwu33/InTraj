from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from .motion_dataset import (
    MotionDataset,
    MotionLaneSegment,
    MotionPolylineFeature,
    MotionScenario,
    MotionTrack,
)


CoordFrame = Literal["local", "global"]
VelocitySource = Literal["finite_difference", "prefer_input"]
HeadingSource = Literal["trajectory_tangent", "prefer_input"]
CropShape = Literal["circle", "square"]


CANONICAL_AGENT_TYPES = (
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
    "unknown",
)
CANONICAL_MAP_TYPES = (
    "lane_centerline",
    "road_line",
    "road_edge",
    "crosswalk",
    "speed_bump",
    "driveway",
    "drivable_area",
)
_AGENT_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(CANONICAL_AGENT_TYPES)}
_MAP_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(CANONICAL_MAP_TYPES)}


@dataclass(frozen=True)
class StandardAgentConfig:
    max_agents: int = 64
    velocity_source: VelocitySource = "finite_difference"
    heading_source: HeadingSource = "trajectory_tangent"
    include_size: bool = False


@dataclass(frozen=True)
class StandardMapConfig:
    range_m: float = 100.0
    precision_m: float = 0.5
    max_polylines: int = 256
    points_per_polyline: int = 20
    crop_shape: CropShape = "circle"
    include_lane_centerlines: bool = True
    include_road_lines: bool = True
    include_road_edges: bool = True
    include_crosswalks: bool = True
    include_speed_bumps: bool = False
    include_driveways: bool = False
    include_drivable_areas: bool = False


@dataclass(frozen=True)
class StandardizationConfig:
    history_steps: int = 40
    future_steps: int = 50
    dt: float = 0.1
    source_current_time_index: int | None = None
    coord_frame: CoordFrame = "local"
    align_heading: bool = True
    agents: StandardAgentConfig = field(default_factory=StandardAgentConfig)
    map: StandardMapConfig = field(default_factory=StandardMapConfig)

    @property
    def total_steps(self) -> int:
        return int(self.history_steps + self.future_steps)

    @property
    def current_index(self) -> int:
        return int(self.history_steps - 1)


StandardConfig = StandardizationConfig()
DEFAULT_STANDARDIZATION_CONFIG = StandardConfig


@dataclass(frozen=True)
class _StandardizedMapFeature:
    feature_id: str
    feature_type: str
    points: np.ndarray
    is_intersection: bool = False


_STANDARDIZATION_METADATA_KEY = "standardization"


def standardize(
    dataset: MotionDataset,
    transform: Callable[[Any], Any] | None = None,
    config: StandardizationConfig | None = None,
) -> MotionDataset:
    if not isinstance(dataset, MotionDataset):
        raise TypeError(f"standardize expects MotionDataset, got {type(dataset)!r}")

    base_transform = dataset._transform

    def _transform(sample: Any) -> Any:
        if base_transform is not None:
            sample = base_transform(sample)
        if not isinstance(sample, MotionScenario):
            raise TypeError(
                "standardize(dataset) expects the input dataset to yield MotionScenario instances"
            )
        standardized = standardize_scenario(sample, config=config)
        if transform is not None:
            return transform(standardized)
        return standardized

    standardized_dataset = MotionDataset(
        scenario_refs=list(dataset.scenario_refs),
        loader=dataset._loader,
        transform=_transform,
    )
    return standardized_dataset


def standardize_scenario(
    scenario: MotionScenario,
    config: StandardizationConfig | None = None,
) -> MotionScenario:
    config = StandardConfig if config is None else config

    current_index, window_start, window_end, window_clamped = _resolve_source_window(
        scenario,
        config,
    )

    anchor_track_id = scenario.sdc_track_id

    # get anchor pose from sdc
    anchor_track = _track_by_id(scenario, anchor_track_id)
    origin = anchor_track.positions[current_index, :2].astype(np.float32)
    theta = anchor_track.headings[current_index] 

    agent_data = _standardize_agents(
        scenario=scenario,
        window_start=window_start,
        window_end=window_end,
        origin=origin,
        theta=theta,
        config=config,
    )
    map_data = _standardize_map(
        scenario=scenario,
        origin=origin,
        theta=theta,
        config=config,
    )

    source_dt = _infer_dt_seconds(scenario.timestamps_seconds)
    metadata = {
        "source_num_steps": int(scenario.num_steps),
        "source_window_start": int(window_start),
        "source_window_end": int(window_end),
        "source_current_time_index": int(current_index),
        "source_current_time_index_requested": (
            None
            if config.source_current_time_index is None
            else int(config.source_current_time_index)
        ),
        "source_window_clamped": bool(window_clamped),

        "source_window_has_padding": bool(
            window_start < 0 or window_end > int(scenario.num_steps)
        ),
        "source_dt_seconds": float(source_dt) if source_dt is not None else None,
        "dt_mismatch": (
            source_dt is not None and not np.isclose(source_dt, config.dt, atol=1e-4)
        ),
        **agent_data["metadata"],
        **map_data["metadata"],
    }

    sdc_track_id = next(
        (track.track_id for track in agent_data["tracks"] if track.is_ego),
        None,
    )

    metadata[_STANDARDIZATION_METADATA_KEY] = {
        "dt": float(config.dt),
        "history_steps": int(config.history_steps),
        "future_steps": int(config.future_steps),
        "coord_frame": config.coord_frame,
        "origin": origin.astype(np.float32),
        "theta": float(theta),
        "anchor_track_id": anchor_track_id,
        "map_points_per_polyline": int(config.map.points_per_polyline),
        "agent_include_size": bool(config.agents.include_size),
        "map_feature_records": tuple(map_data["map_feature_records"]),
    }

    return MotionScenario(
        scenario_id=scenario.scenario_id,
        source=scenario.source,
        split=scenario.split,
        timestamps_seconds=_relative_timestamps(config),
        current_time_index=int(config.history_steps - 1),
        tracks=agent_data["tracks"],
        lane_segments=map_data["lane_segments"],
        road_lines=map_data["road_lines"],
        road_edges=map_data["road_edges"],
        city_name=None,
        focal_track_id=None,
        sdc_track_id=sdc_track_id,
        metadata=metadata,
    )


def is_standardized_scenario(scenario: MotionScenario) -> bool:
    return (
        isinstance(scenario.metadata, dict)
        and _STANDARDIZATION_METADATA_KEY in scenario.metadata
    )


def get_standardization_metadata(scenario: MotionScenario) -> dict[str, Any]:
    metadata = scenario.metadata.get(_STANDARDIZATION_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError(f"Scenario {scenario.scenario_id} is not standardized")
    return metadata


def get_dt(scenario: MotionScenario) -> float:
    return float(get_standardization_metadata(scenario)["dt"])


def get_history_steps(scenario: MotionScenario) -> int:
    return int(get_standardization_metadata(scenario)["history_steps"])


def get_future_steps(scenario: MotionScenario) -> int:
    return int(get_standardization_metadata(scenario)["future_steps"])


def get_coord_frame(scenario: MotionScenario) -> CoordFrame:
    return get_standardization_metadata(scenario)["coord_frame"]


def get_origin(scenario: MotionScenario) -> np.ndarray:
    return np.asarray(
        get_standardization_metadata(scenario)["origin"], dtype=np.float32
    )


def get_theta(scenario: MotionScenario) -> float:
    return float(get_standardization_metadata(scenario)["theta"])


def get_anchor_track_id(scenario: MotionScenario) -> str | None:
    return get_standardization_metadata(scenario)["anchor_track_id"]


def get_relative_timestamps_seconds(scenario: MotionScenario) -> np.ndarray:
    return np.asarray(scenario.timestamps_seconds, dtype=np.float32)


def get_primary_target_track_id(scenario: MotionScenario) -> str | None:
    return scenario.focal_track_id


def get_primary_target_index(scenario: MotionScenario) -> int:
    primary_target_track_id = get_primary_target_track_id(scenario)
    if primary_target_track_id is None:
        return -1
    for idx, track in enumerate(scenario.tracks):
        if track.track_id == primary_target_track_id:
            return int(idx)
    return -1


def get_standardized_agent_arrays(scenario: MotionScenario) -> dict[str, Any]:
    if not is_standardized_scenario(scenario):
        raise ValueError(f"Scenario {scenario.scenario_id} is not standardized")

    include_size = bool(get_standardization_metadata(scenario)["agent_include_size"])
    agent_ids = [track.track_id for track in scenario.tracks]
    agent_types = np.asarray(
        [_canonical_agent_type_index(track.object_type) for track in scenario.tracks],
        dtype=np.int64,
    )
    agent_positions = _stack_track_xy(scenario.tracks, attr_name="positions")
    agent_velocities = _stack_track_xy(scenario.tracks, attr_name="velocities")
    agent_headings = _stack_track_scalar(scenario.tracks, attr_name="headings")
    agent_valid_mask = _stack_track_bool(scenario.tracks, attr_name="valid_mask")
    agent_observed_mask = _stack_track_bool(scenario.tracks, attr_name="observed_mask")
    agent_is_ego = np.asarray([track.is_ego for track in scenario.tracks], dtype=bool)
    agent_is_target = np.asarray(
        [track.is_prediction_target for track in scenario.tracks],
        dtype=bool,
    )
    agent_is_interest = np.asarray(
        [track.is_object_of_interest for track in scenario.tracks],
        dtype=bool,
    )

    agent_size = None
    agent_size_valid_mask = None
    if include_size:
        agent_size = np.zeros((len(scenario.tracks), 3), dtype=np.float32)
        agent_size_valid_mask = np.zeros((len(scenario.tracks),), dtype=bool)
        for idx, track in enumerate(scenario.tracks):
            collapsed = _collapse_track_size(track.sizes, track.valid_mask)
            if collapsed is not None:
                agent_size[idx] = collapsed.astype(np.float32)
                agent_size_valid_mask[idx] = True

    return {
        "agent_ids": agent_ids,
        "agent_types": agent_types,
        "agent_positions": agent_positions,
        "agent_velocities": agent_velocities,
        "agent_headings": agent_headings,
        "agent_valid_mask": agent_valid_mask,
        "agent_observed_mask": agent_observed_mask,
        "agent_is_ego": agent_is_ego,
        "agent_is_target": agent_is_target,
        "agent_is_interest": agent_is_interest,
        "agent_size": agent_size,
        "agent_size_valid_mask": agent_size_valid_mask,
    }


def get_standardized_map_arrays(scenario: MotionScenario) -> dict[str, Any]:
    metadata = get_standardization_metadata(scenario)
    map_feature_records = metadata["map_feature_records"]
    points_per_polyline = int(metadata["map_points_per_polyline"])
    map_ids = [feature.feature_id for feature in map_feature_records]
    map_types = np.asarray(
        [_MAP_TYPE_TO_INDEX[feature.feature_type] for feature in map_feature_records],
        dtype=np.int64,
    )
    map_points = _pad_standardized_map_points(
        map_feature_records,
        points_per_polyline=points_per_polyline,
    )
    map_valid_mask = _pad_standardized_map_valid_mask(
        map_feature_records,
        points_per_polyline=points_per_polyline,
    )
    map_is_intersection = np.asarray(
        [feature.is_intersection for feature in map_feature_records],
        dtype=bool,
    )
    return {
        "map_ids": map_ids,
        "map_types": map_types,
        "map_points": map_points,
        "map_valid_mask": map_valid_mask,
        "map_is_intersection": map_is_intersection,
    }


def _standardize_agents(
    scenario: MotionScenario,
    window_start: int,
    window_end: int,
    origin: np.ndarray,
    theta: float,
    config: StandardizationConfig,
) -> dict[str, Any]:
    max_agents = config.agents.max_agents
    current_index = config.history_steps - 1
    selection_range = config.map.range_m

    prepared_tracks: list[dict[str, Any]] = []
    for track in scenario.tracks:
        
        # 1. crop within window
        cropped = _crop_track(track, window_start, window_end)
        if not cropped["valid_mask"].any():
            continue

        positions = cropped["positions"][:, :2]
        input_velocities = cropped["velocities"][:, :2]
        input_headings = cropped["headings"]

        # 2. transform to local frame
        if config.coord_frame == "local":
            positions = _transform_points(positions, origin, theta)
            input_velocities = _rotate_vectors(input_velocities, theta)
            input_headings = _wrap_angle(input_headings - theta)

        velocities = _standardize_velocities(
            positions=positions,
            input_velocities=input_velocities,
            valid_mask=cropped["valid_mask"],
            dt=config.dt,
            velocity_source=config.agents.velocity_source,
        )
        headings = _standardize_headings(
            positions=positions,
            velocities=velocities,
            input_headings=input_headings,
            valid_mask=cropped["valid_mask"],
            heading_source=config.agents.heading_source,
        )

        reference_position = _reference_position(
            positions,
            cropped["valid_mask"],
            current_index,
        )
        distance = (
            float(np.linalg.norm(reference_position))
            if reference_position is not None
            else float("inf")
        )
        is_interest = bool(track.is_object_of_interest)

        if not (track.is_ego or track.is_prediction_target or is_interest):
            if reference_position is None or distance > selection_range:
                continue

        prepared_tracks.append(
            {
                "track_id": track.track_id,
                "object_type": _canonical_agent_type(track.object_type),
                "category": None,
                "positions": positions.astype(np.float32),
                "velocities": velocities.astype(np.float32),
                "headings": headings.astype(np.float32),
                "valid_mask": cropped["valid_mask"].astype(bool),
                "observed_mask": cropped["observed_mask"].astype(bool),
                "is_ego": bool(track.is_ego),
                "is_target": bool(track.is_prediction_target),
                "is_interest": is_interest,
                "sizes": cropped["sizes"].astype(np.float32),
                "distance": distance,
                "metadata": {
                    "standardized": True,
                    "coord_frame": config.coord_frame,
                    "distance_to_anchor": distance,
                },
                "sort_key": _track_sort_key(
                    track=track,
                    is_interest=is_interest,
                    distance=distance,
                ),
            }
        )

    prepared_tracks.sort(key=lambda item: item["sort_key"])
    omitted_agent_count = max(0, len(prepared_tracks) - max_agents)
    prepared_tracks = prepared_tracks[:max_agents]

    tracks = [
        MotionTrack(
            track_id=item["track_id"],
            object_type=item["object_type"],
            category=item["category"],
            positions=_xy_to_xyz(item["positions"], item["valid_mask"]),
            headings=item["headings"],
            velocities=_xy_to_xyz(
                item["velocities"],
                item["valid_mask"],
                fill_invalid_with_zero=True,
            ),
            sizes=item["sizes"],
            valid_mask=item["valid_mask"],
            observed_mask=item["observed_mask"],
            is_ego=bool(item["is_ego"]),
            is_focal=False,
            is_prediction_target=bool(item["is_target"]),
            is_object_of_interest=bool(item["is_interest"]),
            metadata=item["metadata"],
        )
        for item in prepared_tracks
    ]

    return {
        "tracks": tracks,
        "metadata": {
            "selected_agent_count": int(len(tracks)),
            "omitted_agent_count": int(omitted_agent_count),
        },
    }


def _standardize_map(
    scenario: MotionScenario,
    origin: np.ndarray,
    theta: float,
    config: StandardizationConfig,
) -> dict[str, Any]:
    feature_candidates = _collect_map_feature_candidates(scenario, config.map)
    chunks: list[dict[str, Any]] = []

    for feature in feature_candidates:
        feature_type = feature["feature_type"]
        feature_id = feature["feature_id"]
        points = _extract_xy(feature["polyline"])
        if points.shape[0] < 2:
            continue

        working_points = points
        if config.coord_frame == "local":
            working_points = _transform_points(points, origin, theta)

        resampled = _resample_polyline(working_points, config.map.precision_m)
        if resampled.shape[0] < 2:
            continue

        visible_segments = _crop_polyline_segments(
            resampled,
            range_m=config.map.range_m,
            crop_shape=config.map.crop_shape,
        )
        for segment_idx, segment in enumerate(visible_segments):
            for chunk in _chunk_polyline(
                segment,
                points_per_polyline=config.map.points_per_polyline,
            ):
                if chunk["valid_mask"].sum() < 2:
                    continue
                valid_points = chunk["points"][chunk["valid_mask"]]
                distance = float(np.linalg.norm(valid_points.mean(axis=0)))
                chunks.append(
                    {
                        "feature_id": (
                            feature_id
                            if segment_idx == 0
                            else f"{feature_id}#seg{segment_idx}"
                        ),
                        "feature_type_index": _MAP_TYPE_TO_INDEX[feature_type],
                        "points": chunk["points"].astype(np.float32),
                        "valid_mask": chunk["valid_mask"].astype(bool),
                        "is_intersection": bool(feature["is_intersection"]),
                        "distance": distance,
                        "sort_key": (
                            _MAP_TYPE_TO_INDEX[feature_type],
                            distance,
                            feature_id,
                        ),
                    }
                )

    chunks.sort(key=lambda item: item["sort_key"])
    omitted_polyline_count = max(0, len(chunks) - config.map.max_polylines)
    chunks = chunks[: config.map.max_polylines]

    map_feature_records: list[_StandardizedMapFeature] = []
    lane_segments: list[MotionLaneSegment] = []
    road_lines: list[MotionPolylineFeature] = []
    road_edges: list[MotionPolylineFeature] = []

    for item in chunks:
        valid_points = item["points"][item["valid_mask"]].astype(np.float32)
        if valid_points.shape[0] < 2:
            continue

        feature_type = CANONICAL_MAP_TYPES[int(item["feature_type_index"])]
        map_feature_records.append(
            _StandardizedMapFeature(
                feature_id=item["feature_id"],
                feature_type=feature_type,
                points=valid_points,
                is_intersection=bool(item["is_intersection"]),
            )
        )

        if feature_type == "lane_centerline":
            lane_segments.append(
                MotionLaneSegment(
                    lane_id=item["feature_id"],
                    centerline=valid_points,
                    is_intersection=bool(item["is_intersection"]),
                    metadata={"standardized": True},
                )
            )
        elif feature_type == "road_edge":
            road_edges.append(
                MotionPolylineFeature(
                    feature_id=item["feature_id"],
                    feature_type=feature_type,
                    points=valid_points,
                    metadata={"standardized": True},
                )
            )
        else:
            road_lines.append(
                MotionPolylineFeature(
                    feature_id=item["feature_id"],
                    feature_type=feature_type,
                    points=valid_points,
                    metadata={"standardized": True},
                )
            )

    return {
        "map_feature_records": map_feature_records,
        "lane_segments": lane_segments,
        "road_lines": road_lines,
        "road_edges": road_edges,
        "metadata": {
            "selected_polyline_count": int(len(map_feature_records)),
            "omitted_polyline_count": int(omitted_polyline_count),
        },
    }


def _resolve_track_heading_at_index(track: MotionTrack, index: int, dt: float) -> float:
    if 0 <= index < track.headings.shape[0] and np.isfinite(track.headings[index]):
        return float(track.headings[index])

    positions = track.positions[:, :2]
    velocities = _finite_difference_velocity(positions, track.valid_mask, dt)
    heading = _heading_from_velocity(velocities, track.valid_mask)
    if 0 <= index < heading.shape[0] and np.isfinite(heading[index]):
        return float(heading[index])
    return 0.0


def _track_by_id(
    scenario: MotionScenario,
    track_id: str | None,
) -> MotionTrack | None:
    if track_id is None:
        return None
    return next(
        (track for track in scenario.tracks if track.track_id == track_id),
        None,
    )


def _resolve_source_current_index(
    scenario: MotionScenario,
    config: StandardizationConfig,
) -> int:
    if config.source_current_time_index is not None:
        return int(config.source_current_time_index)
    if scenario.current_time_index is not None:
        return int(scenario.current_time_index)
    return max(int(config.history_steps - 1), 0)


def _resolve_source_window(
    scenario: MotionScenario,
    config: StandardizationConfig,
) -> tuple[int, int, int, bool]:
    preferred_index = _resolve_source_current_index(scenario, config)

    min_current = int(config.history_steps - 1)
    max_current = int(scenario.num_steps - config.future_steps - 1)
    window_clamped = False

    if min_current <= max_current:
        current_index = min(max(preferred_index, min_current), max_current)
        window_clamped = current_index != preferred_index
    else:
        current_index = preferred_index

    window_start = int(current_index - config.history_steps + 1)
    window_end = int(current_index + config.future_steps + 1)
    return int(current_index), window_start, window_end, bool(window_clamped)


def _xy_to_xyz(
    xy_values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fill_invalid_with_zero: bool = False,
) -> np.ndarray:
    xyz = np.zeros((xy_values.shape[0], 3), dtype=np.float32)
    if not fill_invalid_with_zero:
        xyz[:] = np.nan
    xyz[:, :2] = xy_values.astype(np.float32)
    finite_mask = valid_mask & np.isfinite(xy_values).all(axis=-1)
    xyz[finite_mask, 2] = 0.0
    if fill_invalid_with_zero:
        xyz[~finite_mask] = 0.0
    return xyz


def _stack_track_xy(tracks: Sequence[MotionTrack], attr_name: str) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 0, 2), dtype=np.float32)
    return np.stack(
        [
            np.asarray(getattr(track, attr_name), dtype=np.float32)[:, :2]
            for track in tracks
        ],
        axis=0,
    )


def _stack_track_scalar(tracks: Sequence[MotionTrack], attr_name: str) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(
        [np.asarray(getattr(track, attr_name), dtype=np.float32) for track in tracks],
        axis=0,
    )


def _stack_track_bool(tracks: Sequence[MotionTrack], attr_name: str) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 0), dtype=bool)
    return np.stack(
        [np.asarray(getattr(track, attr_name), dtype=bool) for track in tracks],
        axis=0,
    )


def _pad_standardized_map_points(
    map_features: Sequence[_StandardizedMapFeature],
    *,
    points_per_polyline: int,
) -> np.ndarray:
    if not map_features:
        return np.zeros((0, points_per_polyline, 2), dtype=np.float32)

    padded = np.zeros((len(map_features), points_per_polyline, 2), dtype=np.float32)
    for idx, feature in enumerate(map_features):
        count = min(feature.points.shape[0], points_per_polyline)
        padded[idx, :count] = feature.points[:count]
    return padded


def _pad_standardized_map_valid_mask(
    map_features: Sequence[_StandardizedMapFeature],
    *,
    points_per_polyline: int,
) -> np.ndarray:
    if not map_features:
        return np.zeros((0, points_per_polyline), dtype=bool)

    valid_mask = np.zeros((len(map_features), points_per_polyline), dtype=bool)
    for idx, feature in enumerate(map_features):
        count = min(feature.points.shape[0], points_per_polyline)
        valid_mask[idx, :count] = True
    return valid_mask


def collate_standardized_samples(
    batch: list[MotionScenario],
) -> dict[str, Any]:
    if not batch:
        return {}

    agent_arrays = [get_standardized_agent_arrays(item) for item in batch]
    map_arrays = [get_standardized_map_arrays(item) for item in batch]

    collated: dict[str, Any] = {
        "samples": batch,
        "scenario_id": [item.scenario_id for item in batch],
        "source": [item.source for item in batch],
        "split": [item.split for item in batch],
        "city_name": [item.city_name for item in batch],
        "dt": np.asarray([get_dt(item) for item in batch], dtype=np.float32),
        "history_steps": np.asarray(
            [get_history_steps(item) for item in batch], dtype=np.int64
        ),
        "future_steps": np.asarray(
            [get_future_steps(item) for item in batch], dtype=np.int64
        ),
        "current_time_index": np.asarray(
            [item.current_time_index for item in batch],
            dtype=np.int64,
        ),
        "coord_frame": [get_coord_frame(item) for item in batch],
        "origin": _stack_or_list([get_origin(item) for item in batch]),
        "theta": np.asarray([get_theta(item) for item in batch], dtype=np.float32),
        "anchor_track_id": [get_anchor_track_id(item) for item in batch],
        "primary_target_track_id": [
            get_primary_target_track_id(item) for item in batch
        ],
        "primary_target_index": np.asarray(
            [get_primary_target_index(item) for item in batch],
            dtype=np.int64,
        ),
        "timestamps_seconds": _stack_or_list(
            [item.timestamps_seconds for item in batch]
        ),
        "relative_timestamps_seconds": _stack_or_list(
            [get_relative_timestamps_seconds(item) for item in batch]
        ),
        "agent_ids": [item["agent_ids"] for item in agent_arrays],
        "agent_types": _stack_or_list([item["agent_types"] for item in agent_arrays]),
        "agent_positions": _stack_or_list(
            [item["agent_positions"] for item in agent_arrays]
        ),
        "agent_velocities": _stack_or_list(
            [item["agent_velocities"] for item in agent_arrays]
        ),
        "agent_headings": _stack_or_list(
            [item["agent_headings"] for item in agent_arrays]
        ),
        "agent_valid_mask": _stack_or_list(
            [item["agent_valid_mask"] for item in agent_arrays]
        ),
        "agent_observed_mask": _stack_or_list(
            [item["agent_observed_mask"] for item in agent_arrays]
        ),
        "agent_is_ego": _stack_or_list([item["agent_is_ego"] for item in agent_arrays]),
        "agent_is_target": _stack_or_list(
            [item["agent_is_target"] for item in agent_arrays]
        ),
        "agent_is_interest": _stack_or_list(
            [item["agent_is_interest"] for item in agent_arrays]
        ),
        "map_ids": [item["map_ids"] for item in map_arrays],
        "map_types": _stack_or_list([item["map_types"] for item in map_arrays]),
        "map_points": _stack_or_list([item["map_points"] for item in map_arrays]),
        "map_valid_mask": _stack_or_list(
            [item["map_valid_mask"] for item in map_arrays]
        ),
        "map_is_intersection": _stack_or_list(
            [item["map_is_intersection"] for item in map_arrays]
        ),
        "agent_size": _stack_or_list([item["agent_size"] for item in agent_arrays]),
        "agent_size_valid_mask": _stack_or_list(
            [item["agent_size_valid_mask"] for item in agent_arrays]
        ),
        "metadata": [item.metadata for item in batch],
        "tracks": [item.tracks for item in batch],
        "lane_segments": [item.lane_segments for item in batch],
        "road_lines": [item.road_lines for item in batch],
        "road_edges": [item.road_edges for item in batch],
        "agent_type_vocab": CANONICAL_AGENT_TYPES,
        "map_type_vocab": CANONICAL_MAP_TYPES,
    }
    return collated


def _stack_or_list(values: list[Any]) -> Any:
    if not values:
        return values
    if any(value is None for value in values):
        return values if not all(value is None for value in values) else None
    if not all(isinstance(value, np.ndarray) for value in values):
        return values

    shapes = [value.shape for value in values]
    if all(shape == shapes[0] for shape in shapes):
        return np.stack(values, axis=0)
    return values


def _reference_index(valid_mask: np.ndarray, preferred_index: int) -> int | None:
    if valid_mask.size == 0:
        return None
    if 0 <= preferred_index < valid_mask.size and valid_mask[preferred_index]:
        return int(preferred_index)
    earlier = np.flatnonzero(valid_mask[: preferred_index + 1])
    if earlier.size > 0:
        return int(earlier[-1])
    later = np.flatnonzero(valid_mask[preferred_index + 1 :])
    if later.size > 0:
        return int(preferred_index + 1 + later[0])
    return None


def _crop_track(track: MotionTrack, start: int, end: int) -> dict[str, np.ndarray]:
    total_steps = max(end - start, 0)
    positions = np.full(
        (total_steps, track.positions.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    velocities = np.full(
        (total_steps, track.velocities.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    headings = np.full((total_steps,), np.nan, dtype=np.float32)
    sizes = np.full(
        (total_steps, track.sizes.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    valid_mask = np.zeros((total_steps,), dtype=bool)
    observed_mask = np.zeros((total_steps,), dtype=bool)

    src_start = max(start, 0)
    src_end = min(end, track.num_steps)
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        positions[dst_start:dst_end] = track.positions[src_start:src_end]
        velocities[dst_start:dst_end] = track.velocities[src_start:src_end]
        headings[dst_start:dst_end] = track.headings[src_start:src_end]
        sizes[dst_start:dst_end] = track.sizes[src_start:src_end]
        valid_mask[dst_start:dst_end] = track.valid_mask[src_start:src_end]
        observed_mask[dst_start:dst_end] = track.observed_mask[src_start:src_end]

    return {
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "sizes": sizes,
        "valid_mask": valid_mask,
        "observed_mask": observed_mask,
    }


def _transform_points(
    points: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray:
    transformed = np.full_like(points, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(points).all(axis=-1)
    if not finite_mask.any():
        return transformed
    centered = points[finite_mask] - origin[None, :]
    transformed[finite_mask] = _rotate_xy(centered, theta)
    return transformed


def _rotate_vectors(vectors: np.ndarray, theta: float) -> np.ndarray:
    rotated = np.full_like(vectors, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(vectors).all(axis=-1)
    if not finite_mask.any():
        return rotated
    rotated[finite_mask] = _rotate_xy(vectors[finite_mask], theta)
    return rotated


def _rotate_xy(points: np.ndarray, theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    return points @ rot.T


def _standardize_velocities(
    positions: np.ndarray,
    input_velocities: np.ndarray,
    valid_mask: np.ndarray,
    dt: float,
    velocity_source: VelocitySource,
) -> np.ndarray:
    fd_velocity = _finite_difference_velocity(positions, valid_mask, dt)
    if velocity_source == "finite_difference":
        return fd_velocity.astype(np.float32)

    output = fd_velocity.copy()
    finite_input = np.isfinite(input_velocities).all(axis=-1)
    output[finite_input] = input_velocities[finite_input]
    return output.astype(np.float32)


def _standardize_headings(
    positions: np.ndarray,
    velocities: np.ndarray,
    input_headings: np.ndarray,
    valid_mask: np.ndarray,
    heading_source: HeadingSource,
) -> np.ndarray:
    tangent_heading = _heading_from_velocity(velocities, valid_mask)
    if heading_source == "trajectory_tangent":
        return tangent_heading.astype(np.float32)

    output = tangent_heading.copy()
    finite_input = np.isfinite(input_headings)
    output[finite_input] = _wrap_angle(input_headings[finite_input])
    return output.astype(np.float32)


def _finite_difference_velocity(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    dt: float,
) -> np.ndarray:
    velocities = np.zeros_like(positions, dtype=np.float32)
    finite_mask = valid_mask & np.isfinite(positions).all(axis=-1)

    for idx in range(1, positions.shape[0]):
        if finite_mask[idx] and finite_mask[idx - 1]:
            velocities[idx] = (positions[idx] - positions[idx - 1]) / dt

    for idx in range(positions.shape[0] - 1):
        if (
            finite_mask[idx]
            and finite_mask[idx + 1]
            and np.allclose(velocities[idx], 0.0)
        ):
            velocities[idx] = velocities[idx + 1]

    velocities[~finite_mask] = 0.0
    return velocities


def _heading_from_velocity(
    velocities: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    headings = np.zeros((velocities.shape[0],), dtype=np.float32)
    speed = np.linalg.norm(velocities, axis=-1)
    finite_heading = valid_mask & np.isfinite(speed) & (speed > 1e-4)
    headings[finite_heading] = np.arctan2(
        velocities[finite_heading, 1],
        velocities[finite_heading, 0],
    )

    last_heading = 0.0
    for idx in range(headings.shape[0]):
        if finite_heading[idx]:
            last_heading = float(headings[idx])
        else:
            headings[idx] = last_heading

    return _wrap_angle(headings)


def _reference_position(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    preferred_index: int,
) -> np.ndarray | None:
    index = _reference_index(valid_mask, preferred_index)
    if index is None:
        return None
    return positions[index]


def _collapse_track_size(
    sizes: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray | None:
    finite_mask = valid_mask & np.isfinite(sizes).all(axis=-1)
    if not finite_mask.any():
        return None
    return sizes[np.flatnonzero(finite_mask)[-1]]


def _track_sort_key(
    track: MotionTrack,
    is_interest: bool,
    distance: float,
) -> tuple[int, float, str]:
    if track.is_ego:
        return (0, distance, track.track_id)
    if track.is_prediction_target:
        return (1, distance, track.track_id)
    if is_interest:
        return (2, distance, track.track_id)
    return (3, distance, track.track_id)


def _collect_map_feature_candidates(
    scenario: MotionScenario,
    config: StandardMapConfig,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if config.include_lane_centerlines:
        candidates.extend(
            {
                "feature_type": "lane_centerline",
                "feature_id": lane.lane_id,
                "polyline": lane.centerline,
                "is_intersection": bool(lane.is_intersection),
            }
            for lane in scenario.lane_segments
        )
    if config.include_road_lines:
        candidates.extend(
            {
                "feature_type": "road_line",
                "feature_id": feature.feature_id,
                "polyline": feature.points,
                "is_intersection": False,
            }
            for feature in scenario.road_lines
        )
    if config.include_road_edges:
        candidates.extend(
            {
                "feature_type": "road_edge",
                "feature_id": feature.feature_id,
                "polyline": feature.points,
                "is_intersection": False,
            }
            for feature in scenario.road_edges
        )
    if config.include_crosswalks:
        candidates.extend(
            {
                "feature_type": "crosswalk",
                "feature_id": feature.feature_id,
                "polyline": _polygon_to_boundary(feature.polygon),
                "is_intersection": False,
            }
            for feature in scenario.crosswalks
        )
    if config.include_speed_bumps:
        candidates.extend(
            {
                "feature_type": "speed_bump",
                "feature_id": feature.feature_id,
                "polyline": _polygon_to_boundary(feature.polygon),
                "is_intersection": False,
            }
            for feature in scenario.speed_bumps
        )
    if config.include_driveways:
        candidates.extend(
            {
                "feature_type": "driveway",
                "feature_id": feature.feature_id,
                "polyline": _polygon_to_boundary(feature.polygon),
                "is_intersection": False,
            }
            for feature in scenario.driveways
        )
    if config.include_drivable_areas:
        candidates.extend(
            {
                "feature_type": "drivable_area",
                "feature_id": feature.feature_id,
                "polyline": _polygon_to_boundary(feature.polygon),
                "is_intersection": False,
            }
            for feature in scenario.drivable_areas
        )
    return candidates


def _extract_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected [N, D] polyline array, got shape {points.shape}")
    return points[:, :2]


def _polygon_to_boundary(polygon: np.ndarray) -> np.ndarray:
    polygon_xy = _extract_xy(polygon)
    if polygon_xy.shape[0] == 0:
        return polygon_xy
    if not np.allclose(polygon_xy[0], polygon_xy[-1]):
        polygon_xy = np.concatenate([polygon_xy, polygon_xy[:1]], axis=0)
    return polygon_xy


def _resample_polyline(points: np.ndarray, precision_m: float) -> np.ndarray:
    points = _clean_polyline(points)
    if points.shape[0] < 2:
        return points
    if precision_m <= 0:
        return points

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=-1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(seg_lengths, dtype=np.float32)]
    )
    total_length = float(cumulative[-1])
    if total_length < 1e-6:
        return points[:1]

    sample_s = np.arange(0.0, total_length, precision_m, dtype=np.float32)
    if sample_s.size == 0 or sample_s[-1] < total_length:
        sample_s = np.concatenate(
            [sample_s, np.asarray([total_length], dtype=np.float32)]
        )

    x = np.interp(sample_s, cumulative, points[:, 0])
    y = np.interp(sample_s, cumulative, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def _clean_polyline(points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    points = points[finite]
    if points.shape[0] < 2:
        return points.astype(np.float32)

    deduped = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, deduped[-1]):
            deduped.append(point)
    return np.asarray(deduped, dtype=np.float32)


def _crop_polyline_segments(
    points: np.ndarray,
    range_m: float,
    crop_shape: CropShape,
) -> list[np.ndarray]:
    if points.shape[0] < 2:
        return []

    if crop_shape == "circle":
        inside = np.linalg.norm(points, axis=-1) <= range_m
    elif crop_shape == "square":
        inside = (np.abs(points[:, 0]) <= range_m) & (np.abs(points[:, 1]) <= range_m)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported crop_shape: {crop_shape}")

    segments: list[np.ndarray] = []
    start_idx = None
    for idx, keep in enumerate(inside):
        if keep and start_idx is None:
            start_idx = idx
        elif not keep and start_idx is not None:
            if idx - start_idx >= 2:
                segments.append(points[start_idx:idx])
            start_idx = None
    if start_idx is not None and points.shape[0] - start_idx >= 2:
        segments.append(points[start_idx:])
    return segments


def _chunk_polyline(
    points: np.ndarray,
    points_per_polyline: int,
) -> list[dict[str, np.ndarray]]:
    if points.shape[0] < 2:
        return []
    if points_per_polyline <= 1:
        raise ValueError("points_per_polyline must be > 1")

    stride = max(points_per_polyline - 1, 1)
    chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, points.shape[0], stride):
        chunk_points = points[start : start + points_per_polyline]
        if chunk_points.shape[0] < 2:
            break
        padded = np.zeros((points_per_polyline, 2), dtype=np.float32)
        valid_mask = np.zeros((points_per_polyline,), dtype=bool)
        padded[: chunk_points.shape[0]] = chunk_points
        valid_mask[: chunk_points.shape[0]] = True
        chunks.append({"points": padded, "valid_mask": valid_mask})
        if start + points_per_polyline >= points.shape[0]:
            break
    return chunks


def _infer_dt_seconds(timestamps_seconds: np.ndarray) -> float | None:
    if timestamps_seconds.shape[0] < 2:
        return None
    diffs = np.diff(timestamps_seconds.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _relative_timestamps(config: StandardizationConfig) -> np.ndarray:
    history = np.arange(-config.history_steps + 1, 1, dtype=np.float32)
    future = np.arange(1, config.future_steps + 1, dtype=np.float32)
    return np.concatenate([history, future], axis=0) * float(config.dt)


def _canonical_agent_type_index(raw_type: str) -> int:
    canonical = _canonical_agent_type(raw_type)
    return _AGENT_TYPE_TO_INDEX.get(canonical, _AGENT_TYPE_TO_INDEX["unknown"])


def _canonical_agent_type(raw_type: str) -> str:
    raw = raw_type.lower().strip()
    if raw.startswith("type_"):
        raw = raw[5:]
    aliases = {
        "vehicle": "vehicle",
        "pedestrian": "pedestrian",
        "cyclist": "cyclist",
        "bicyclist": "cyclist",
        "motorcyclist": "motorcyclist",
        "motorcycle": "motorcyclist",
        "bus": "bus",
        "static": "static",
        "background": "background",
        "construction": "construction",
        "riderless_bicycle": "riderless_bicycle",
        "other": "unknown",
        "unknown": "unknown",
    }
    return aliases.get(raw, "unknown")


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float32)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "CANONICAL_AGENT_TYPES",
    "CANONICAL_MAP_TYPES",
    "DEFAULT_STANDARDIZATION_CONFIG",
    "StandardAgentConfig",
    "StandardConfig",
    "StandardMapConfig",
    "StandardizationConfig",
    "collate_standardized_samples",
    "get_anchor_track_id",
    "get_coord_frame",
    "get_dt",
    "get_future_steps",
    "get_history_steps",
    "get_origin",
    "get_primary_target_index",
    "get_primary_target_track_id",
    "get_relative_timestamps_seconds",
    "get_standardization_metadata",
    "get_standardized_agent_arrays",
    "get_standardized_map_arrays",
    "get_theta",
    "is_standardized_scenario",
    "resolve_standardization_config",
    "standardize",
    "standardize_scenario",
]
