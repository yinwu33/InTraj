from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from .motion_dataset import MotionScenario
from .standardization import (
    StandardizationConfig,
    get_primary_target_index,
    get_primary_target_track_id,
    get_standardization_metadata,
    get_standardized_agent_arrays,
    get_standardized_map_arrays,
    standardize_scenario,
)


def normalize_split_name(split: str | None) -> str:
    if split is None:
        raise ValueError("split cannot be None for standardized cache paths")

    normalized = str(split).lower()
    if normalized == "training":
        return "train"
    if normalized == "validation":
        return "val"
    if normalized == "testing":
        return "test"
    if normalized in {"train", "val", "test"}:
        return normalized
    raise ValueError(f"Unsupported split name for standardized cache: {split}")


def normalize_source_name(source: str) -> str:
    normalized = str(source).lower()
    if normalized not in {"av2", "waymo"}:
        raise ValueError(f"Unsupported standardized cache source: {source}")
    return normalized


def get_standardized_cache_path(
    cache_root: str | Path,
    *,
    source: str,
    split: str,
    scenario_id: str,
) -> Path:
    root = Path(cache_root).expanduser()
    return (
        root
        / normalize_source_name(source)
        / normalize_split_name(split)
        / f"{scenario_id}.pt"
    )


def build_standardized_record(
    scenario: MotionScenario,
    *,
    config: StandardizationConfig,
    split: str | None = None,
) -> dict[str, Any]:
    standardized = standardize_scenario(scenario, config=config)
    agent_arrays = get_standardized_agent_arrays(standardized)
    map_arrays = get_standardized_map_arrays(standardized)
    standardization_metadata = dict(get_standardization_metadata(standardized))
    standardization_metadata.pop("map_feature_records", None)
    metadata = dict(standardized.metadata)
    if isinstance(metadata.get("standardization"), dict):
        metadata["standardization"] = dict(metadata["standardization"])
        metadata["standardization"].pop("map_feature_records", None)

    return {
        "schema_version": 1,
        "scenario_id": standardized.scenario_id,
        "source": normalize_source_name(standardized.source),
        "split": normalize_split_name(split or standardized.split),
        "city_name": standardized.city_name,
        "timestamps_seconds": standardized.timestamps_seconds,
        "current_time_index": standardized.current_time_index,
        "standardization_config": asdict(config),
        "standardization_metadata": standardization_metadata,
        "primary_target_track_id": get_primary_target_track_id(standardized),
        "primary_target_index": get_primary_target_index(standardized),
        "agent_ids": agent_arrays["agent_ids"],
        "agent_types": agent_arrays["agent_types"],
        "agent_positions": agent_arrays["agent_positions"],
        "agent_velocities": agent_arrays["agent_velocities"],
        "agent_headings": agent_arrays["agent_headings"],
        "agent_valid_mask": agent_arrays["agent_valid_mask"],
        "agent_observed_mask": agent_arrays["agent_observed_mask"],
        "agent_is_ego": agent_arrays["agent_is_ego"],
        "agent_is_target": agent_arrays["agent_is_target"],
        "agent_is_interest": agent_arrays["agent_is_interest"],
        "agent_size": agent_arrays["agent_size"],
        "agent_size_valid_mask": agent_arrays["agent_size_valid_mask"],
        "map_ids": map_arrays["map_ids"],
        "map_types": map_arrays["map_types"],
        "map_points": map_arrays["map_points"],
        "map_valid_mask": map_arrays["map_valid_mask"],
        "map_is_intersection": map_arrays["map_is_intersection"],
        "metadata": metadata,
    }


def save_standardized_record(path: str | Path, record: dict[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        prefix=f".{target_path.stem}.",
        suffix=".tmp",
        dir=target_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    try:
        torch.save(record, temp_path)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_standardized_record(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


__all__ = [
    "build_standardized_record",
    "get_standardized_cache_path",
    "load_standardized_record",
    "normalize_source_name",
    "normalize_split_name",
    "save_standardized_record",
]
