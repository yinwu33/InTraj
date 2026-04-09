"""Analyze AV2 ego/focus motion statistics and simple ODD context metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point
from shapely.geometry import Polygon

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


DT_SECONDS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AV2 ego/focus distributions from raw scenario files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory containing AV2 split folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyze, for example train or val.",
    )
    parser.add_argument(
        "--observed-steps",
        type=int,
        default=50,
        help="Number of observed steps to analyze. Default matches AV2 history length.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of scenarios to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Defaults to outputs/av2_analysis/<split>.",
    )
    parser.add_argument(
        "--intersection-near-threshold-m",
        type=float,
        default=20.0,
        help="Distance threshold for flagging a track as intersection-near.",
    )
    parser.add_argument(
        "--crosswalk-near-threshold-m",
        type=float,
        default=10.0,
        help="Distance threshold for flagging a track as crosswalk-near.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Log progress every N processed scenarios.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Persist intermediate outputs every N processed scenarios.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    return parser.parse_args()


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def scenario_dirs(data_root: Path, split: str) -> list[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    return dirs


def load_scenario(scenario_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_id = scenario_dir.name
    parquet_path = scenario_dir / f"scenario_{scenario_id}.parquet"
    map_matches = sorted(scenario_dir.glob("log_map_archive_*.json"))
    if not parquet_path.exists():
        raise FileNotFoundError(f"Scenario parquet not found: {parquet_path}")
    if not map_matches:
        raise FileNotFoundError(f"Map json not found under {scenario_dir}")
    df = pd.read_parquet(parquet_path)
    with map_matches[0].open() as handle:
        map_data = json.load(handle)
    return df, map_data


def extract_track(df: pd.DataFrame, track_id: str, observed_steps: int) -> pd.DataFrame | None:
    track = (
        df[(df["track_id"].astype(str) == str(track_id)) & (df["timestep"] < observed_steps)]
        .sort_values("timestep")
        .drop_duplicates(subset="timestep", keep="last")
        .copy()
    )
    if track.empty:
        return None
    track = track.set_index("timestep").reindex(range(observed_steps))
    track.index.name = "timestep"
    return track.reset_index()


def finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1)


def compute_fd_velocity(positions: np.ndarray) -> np.ndarray:
    fd_velocity = np.full_like(positions, np.nan, dtype=float)
    valid = finite_mask(positions)
    for idx in range(1, len(positions)):
        if valid[idx] and valid[idx - 1]:
            fd_velocity[idx] = (positions[idx] - positions[idx - 1]) / DT_SECONDS
    if len(positions) > 1 and np.isnan(fd_velocity[0]).all() and np.isfinite(fd_velocity[1]).all():
        fd_velocity[0] = fd_velocity[1]
    return fd_velocity


def choose_velocity_vectors(raw_velocity: np.ndarray, fd_velocity: np.ndarray) -> tuple[np.ndarray, str]:
    raw_speed = np.linalg.norm(raw_velocity, axis=1)
    fd_speed = np.linalg.norm(fd_velocity, axis=1)
    raw_valid = np.isfinite(raw_speed)
    fd_valid = np.isfinite(fd_speed)
    raw_median = float(np.nanmedian(raw_speed[raw_valid])) if raw_valid.any() else np.nan
    fd_median = float(np.nanmedian(fd_speed[fd_valid])) if fd_valid.any() else np.nan

    use_fd = False
    if not raw_valid.any() and fd_valid.any():
        use_fd = True
    elif raw_valid.any() and fd_valid.any() and raw_median < 1e-3 and fd_median > 0.5:
        use_fd = True

    if use_fd:
        return fd_velocity, "finite_diff"

    velocity = raw_velocity.copy()
    missing_raw = ~np.isfinite(velocity).all(axis=1)
    velocity[missing_raw] = fd_velocity[missing_raw]
    return velocity, "raw"


def nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def path_length(positions: np.ndarray) -> float:
    valid = finite_mask(positions)
    length = 0.0
    for idx in range(1, len(positions)):
        if valid[idx] and valid[idx - 1]:
            length += float(np.linalg.norm(positions[idx] - positions[idx - 1]))
    return length


def heading_change_abs_sum(heading: np.ndarray) -> float:
    finite = np.isfinite(heading)
    if finite.sum() < 2:
        return float("nan")
    diffs = wrap_angle(np.diff(heading[finite]))
    return float(np.abs(diffs).sum())


def motion_metrics(track: pd.DataFrame, prefix: str) -> dict[str, Any]:
    positions = track[["position_x", "position_y"]].to_numpy(dtype=float)
    raw_velocity = track[["velocity_x", "velocity_y"]].to_numpy(dtype=float)
    fd_velocity = compute_fd_velocity(positions)
    velocity, velocity_source = choose_velocity_vectors(raw_velocity, fd_velocity)
    speed = np.linalg.norm(velocity, axis=1)
    heading = track["heading"].to_numpy(dtype=float)
    valid = finite_mask(positions)

    return {
        f"{prefix}_observed_ratio": float(valid.mean()),
        f"{prefix}_speed_source": velocity_source,
        f"{prefix}_speed_min_mps": float(np.nanmin(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_mean_mps": float(np.nanmean(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_p95_mps": nanpercentile(speed, 95),
        f"{prefix}_speed_max_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan"),
        f"{prefix}_speed_last_mps": last_finite(speed),
        f"{prefix}_path_length_obs_m": path_length(positions),
        f"{prefix}_heading_change_abs_sum_rad": heading_change_abs_sum(heading),
        f"{prefix}_num_observed_steps": int(valid.sum()),
        f"{prefix}_last_x": last_finite(positions[:, 0]),
        f"{prefix}_last_y": last_finite(positions[:, 1]),
    }, positions, velocity


def compute_interaction_metrics(
    ego_positions: np.ndarray,
    ego_velocity: np.ndarray,
    focus_positions: np.ndarray,
    focus_velocity: np.ndarray,
) -> dict[str, Any]:
    valid = finite_mask(ego_positions) & finite_mask(focus_positions)
    if not valid.any():
        return {
            "distance_last_m": float("nan"),
            "min_distance_obs_m": float("nan"),
            "min_distance_timestep": float("nan"),
            "relative_speed_last_mps": float("nan"),
            "closing_speed_at_min_distance_mps": float("nan"),
            "num_overlap_steps": 0,
        }

    separation = focus_positions - ego_positions
    distance = np.linalg.norm(separation, axis=1)
    overlap_steps = np.flatnonzero(valid)
    valid_distance = distance[valid]
    min_idx_within_valid = int(np.argmin(valid_distance))
    min_timestep = int(overlap_steps[min_idx_within_valid])
    min_distance = float(valid_distance[min_idx_within_valid])

    last_overlap_timestep = int(overlap_steps[-1])
    relative_velocity = focus_velocity - ego_velocity
    relative_speed_last = (
        float(np.linalg.norm(relative_velocity[last_overlap_timestep]))
        if np.isfinite(relative_velocity[last_overlap_timestep]).all()
        else float("nan")
    )

    closing_speed = float("nan")
    rel_velocity_at_min = relative_velocity[min_timestep]
    separation_at_min = separation[min_timestep]
    if np.isfinite(rel_velocity_at_min).all() and np.isfinite(separation_at_min).all() and min_distance > 1e-6:
        line_of_sight = separation_at_min / min_distance
        closing_speed = float(-np.dot(rel_velocity_at_min, line_of_sight))

    return {
        "distance_last_m": float(distance[last_overlap_timestep]),
        "min_distance_obs_m": min_distance,
        "min_distance_timestep": min_timestep,
        "relative_speed_last_mps": relative_speed_last,
        "closing_speed_at_min_distance_mps": closing_speed,
        "num_overlap_steps": int(valid.sum()),
    }


def coords2d(points: list[dict[str, float]]) -> list[tuple[float, float]]:
    return [(float(point["x"]), float(point["y"])) for point in points]


def build_map_geometries(map_data: dict[str, Any]) -> dict[str, Any]:
    lane_lines: list[LineString] = []
    intersection_lane_lines: list[LineString] = []
    crosswalk_polygons: list[Polygon] = []

    for lane in map_data.get("lane_segments", {}).values():
        centerline = coords2d(lane["centerline"])
        if len(centerline) < 2:
            continue
        line = LineString(centerline)
        lane_lines.append(line)
        if lane.get("is_intersection") is True:
            intersection_lane_lines.append(line)

    for crosswalk in map_data.get("pedestrian_crossings", {}).values():
        edge1 = coords2d(crosswalk["edge1"])
        edge2 = coords2d(crosswalk["edge2"])
        polygon_points = edge1 + list(reversed(edge2))
        if len(polygon_points) < 3:
            continue
        polygon = Polygon(polygon_points)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        crosswalk_polygons.append(polygon)

    lane_geometry = MultiLineString(lane_lines) if lane_lines else GeometryCollection()
    intersection_geometry = (
        MultiLineString(intersection_lane_lines) if intersection_lane_lines else GeometryCollection()
    )
    crosswalk_geometry = (
        GeometryCollection(crosswalk_polygons) if crosswalk_polygons else GeometryCollection()
    )
    return {
        "lane_geometry": lane_geometry,
        "intersection_geometry": intersection_geometry,
        "crosswalk_geometry": crosswalk_geometry,
        "num_lane_segments": len(lane_lines),
        "num_intersection_lane_segments": len(intersection_lane_lines),
        "num_crosswalks": len(crosswalk_polygons),
        "num_drivable_areas": len(map_data.get("drivable_areas", {})),
    }


def min_track_distance_to_geometry(positions: np.ndarray, geometry: Any) -> float:
    if geometry.is_empty:
        return float("nan")
    valid_positions = positions[finite_mask(positions)]
    if valid_positions.size == 0:
        return float("nan")
    return float(min(Point(x, y).distance(geometry) for x, y in valid_positions))


def compute_map_metrics(
    ego_positions: np.ndarray,
    focus_positions: np.ndarray,
    map_geometries: dict[str, Any],
    intersection_threshold_m: float,
    crosswalk_threshold_m: float,
) -> dict[str, Any]:
    ego_lane_distance = min_track_distance_to_geometry(ego_positions, map_geometries["lane_geometry"])
    focus_lane_distance = min_track_distance_to_geometry(focus_positions, map_geometries["lane_geometry"])
    ego_intersection_distance = min_track_distance_to_geometry(
        ego_positions, map_geometries["intersection_geometry"]
    )
    focus_intersection_distance = min_track_distance_to_geometry(
        focus_positions, map_geometries["intersection_geometry"]
    )
    ego_crosswalk_distance = min_track_distance_to_geometry(
        ego_positions, map_geometries["crosswalk_geometry"]
    )
    focus_crosswalk_distance = min_track_distance_to_geometry(
        focus_positions, map_geometries["crosswalk_geometry"]
    )

    return {
        "num_lane_segments": map_geometries["num_lane_segments"],
        "num_intersection_lane_segments": map_geometries["num_intersection_lane_segments"],
        "num_crosswalks": map_geometries["num_crosswalks"],
        "num_drivable_areas": map_geometries["num_drivable_areas"],
        "ego_nearest_lane_centerline_distance_m": ego_lane_distance,
        "focus_nearest_lane_centerline_distance_m": focus_lane_distance,
        "ego_nearest_intersection_lane_distance_m": ego_intersection_distance,
        "focus_nearest_intersection_lane_distance_m": focus_intersection_distance,
        "ego_nearest_crosswalk_distance_m": ego_crosswalk_distance,
        "focus_nearest_crosswalk_distance_m": focus_crosswalk_distance,
        "ego_intersection_near": bool(
            np.isfinite(ego_intersection_distance) and ego_intersection_distance <= intersection_threshold_m
        ),
        "focus_intersection_near": bool(
            np.isfinite(focus_intersection_distance) and focus_intersection_distance <= intersection_threshold_m
        ),
        "ego_crosswalk_near": bool(
            np.isfinite(ego_crosswalk_distance) and ego_crosswalk_distance <= crosswalk_threshold_m
        ),
        "focus_crosswalk_near": bool(
            np.isfinite(focus_crosswalk_distance) and focus_crosswalk_distance <= crosswalk_threshold_m
        ),
    }


def analyze_scenario(
    scenario_dir: Path,
    observed_steps: int,
    intersection_threshold_m: float,
    crosswalk_threshold_m: float,
) -> dict[str, Any]:
    df, map_data = load_scenario(scenario_dir)
    scenario_id = str(df["scenario_id"].iloc[0])
    city = str(df["city"].iloc[0])
    focal_track_id = str(df["focal_track_id"].iloc[0])

    ego_track = extract_track(df, "AV", observed_steps)
    if ego_track is None:
        raise ValueError(f"Scenario {scenario_id} is missing AV track.")

    focus_track = extract_track(df, focal_track_id, observed_steps)
    if focus_track is None:
        raise ValueError(f"Scenario {scenario_id} is missing focal track {focal_track_id}.")

    ego_type = str(ego_track["object_type"].dropna().iloc[0]) if ego_track["object_type"].notna().any() else "unknown"
    focus_type = (
        str(focus_track["object_type"].dropna().iloc[0]) if focus_track["object_type"].notna().any() else "unknown"
    )

    scenario_metrics: dict[str, Any] = {
        "scenario_id": scenario_id,
        "city": city,
        "focal_track_id": focal_track_id,
        "ego_track_id": "AV",
        "ego_type": ego_type,
        "focus_type": focus_type,
    }

    ego_motion, ego_positions, ego_velocity = motion_metrics(ego_track, "ego")
    focus_motion, focus_positions, focus_velocity = motion_metrics(focus_track, "focus")
    scenario_metrics.update(ego_motion)
    scenario_metrics.update(focus_motion)
    scenario_metrics.update(
        compute_interaction_metrics(ego_positions, ego_velocity, focus_positions, focus_velocity)
    )

    map_geometries = build_map_geometries(map_data)
    scenario_metrics.update(
        compute_map_metrics(
            ego_positions,
            focus_positions,
            map_geometries,
            intersection_threshold_m=intersection_threshold_m,
            crosswalk_threshold_m=crosswalk_threshold_m,
        )
    )
    return scenario_metrics


def describe_numeric(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def build_summary(results: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        "ego_speed_p95_mps",
        "ego_speed_max_mps",
        "ego_speed_last_mps",
        "focus_speed_p95_mps",
        "focus_speed_max_mps",
        "focus_speed_last_mps",
        "distance_last_m",
        "min_distance_obs_m",
        "relative_speed_last_mps",
        "closing_speed_at_min_distance_mps",
        "ego_nearest_lane_centerline_distance_m",
        "focus_nearest_lane_centerline_distance_m",
        "ego_nearest_intersection_lane_distance_m",
        "focus_nearest_intersection_lane_distance_m",
        "ego_nearest_crosswalk_distance_m",
        "focus_nearest_crosswalk_distance_m",
    ]
    bool_columns = [
        "ego_intersection_near",
        "focus_intersection_near",
        "ego_crosswalk_near",
        "focus_crosswalk_near",
    ]

    summary: dict[str, Any] = {
        "num_scenarios": int(len(results)),
        "city_counts": {str(key): int(value) for key, value in results["city"].value_counts().items()},
        "ego_type_counts": {
            str(key): int(value) for key, value in results["ego_type"].value_counts(dropna=False).items()
        },
        "focus_type_counts": {
            str(key): int(value) for key, value in results["focus_type"].value_counts(dropna=False).items()
        },
        "numeric_metrics": {},
        "boolean_metrics": {},
    }

    for column in numeric_columns:
        if column in results.columns:
            summary["numeric_metrics"][column] = describe_numeric(results[column])

    for column in bool_columns:
        if column in results.columns:
            summary["boolean_metrics"][column] = {
                "count_true": int(results[column].sum()),
                "share_true": float(results[column].mean()),
            }

    closest = results.nsmallest(20, "min_distance_obs_m")[
        ["scenario_id", "city", "min_distance_obs_m", "distance_last_m"]
    ]
    summary["closest_scenarios"] = closest.to_dict(orient="records")
    return summary


def save_plots(results: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        print("matplotlib not available, skipping plots.")
        return

    plot_specs = [
        ("ego_speed_p95_mps", "Ego speed p95 (m/s)", "hist_ego_speed_p95_mps.png"),
        ("focus_speed_p95_mps", "Focus speed p95 (m/s)", "hist_focus_speed_p95_mps.png"),
        ("min_distance_obs_m", "Min ego-focus distance in history (m)", "hist_min_distance_obs_m.png"),
    ]

    for column, title, filename in plot_specs:
        if column not in results.columns:
            continue
        values = results[column].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    scatter_columns = ["ego_speed_p95_mps", "focus_speed_p95_mps", "min_distance_obs_m"]
    if all(column in results.columns for column in scatter_columns):
        scatter_df = results[scatter_columns].dropna()
        if not scatter_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                scatter_df["ego_speed_p95_mps"],
                scatter_df["focus_speed_p95_mps"],
                s=8,
                alpha=0.4,
            )
            ax.set_xlabel("ego_speed_p95_mps")
            ax.set_ylabel("focus_speed_p95_mps")
            ax.set_title("Ego vs focus speed p95")
            fig.tight_layout()
            fig.savefig(output_dir / "scatter_ego_vs_focus_speed_p95_mps.png", dpi=150)
            plt.close(fig)


def flush_outputs(
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    output_dir: Path,
    split: str,
    *,
    include_plots: bool,
) -> tuple[Path, Path, Path, Path]:
    csv_path = output_dir / f"{split}_ego_focus_metrics.csv"
    parquet_path = output_dir / f"{split}_ego_focus_metrics.parquet"
    summary_path = output_dir / f"{split}_summary.json"
    failures_path = output_dir / f"{split}_failures.json"

    results_df = pd.DataFrame(results).sort_values("scenario_id").reset_index(drop=True)
    summary = build_summary(results_df)

    results_df.to_csv(csv_path, index=False)
    results_df.to_parquet(parquet_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    failures_path.write_text(json.dumps(failures, indent=2))

    if include_plots:
        save_plots(results_df, output_dir)

    return csv_path, parquet_path, summary_path, failures_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path("outputs") / "av2_analysis" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = scenario_dirs(args.data_root, args.split)
    if args.limit is not None:
        dirs = dirs[: args.limit]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    print(
        f"Analyzing split={args.split} scenarios={len(dirs)} observed_steps={args.observed_steps} "
        f"output_dir={output_dir}"
    )

    for index, scenario_dir in enumerate(dirs, start=1):
        try:
            metrics = analyze_scenario(
                scenario_dir,
                observed_steps=args.observed_steps,
                intersection_threshold_m=args.intersection_near_threshold_m,
                crosswalk_threshold_m=args.crosswalk_near_threshold_m,
            )
            results.append(metrics)
        except Exception as exc:  # pragma: no cover - depends on data quality
            failures.append({"scenario_dir": str(scenario_dir), "error": str(exc)})

        if index % max(args.progress_every, 1) == 0 or index == len(dirs):
            print(
                f"processed={index}/{len(dirs)} success={len(results)} failed={len(failures)}"
            )

        if (
            index % max(args.save_every, 1) == 0
            or index == len(dirs)
            or (failures and len(failures) % max(args.save_every, 1) == 0)
        ):
            if results:
                csv_path, parquet_path, summary_path, failures_path = flush_outputs(
                    results,
                    failures,
                    output_dir,
                    args.split,
                    include_plots=False,
                )
                print(
                    "checkpoint saved: "
                    f"processed={index}/{len(dirs)} "
                    f"csv={csv_path.name} parquet={parquet_path.name} "
                    f"summary={summary_path.name} failures={failures_path.name}"
                )
            else:
                failures_path = output_dir / f"{args.split}_failures.json"
                failures_path.write_text(json.dumps(failures, indent=2))
                print(
                    "checkpoint saved: "
                    f"processed={index}/{len(dirs)} failures={failures_path.name}"
                )

    if not results:
        print("No scenarios were successfully analyzed.")
        failures_path = output_dir / f"{args.split}_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2))
        print(f"Saved failures to {failures_path}")
        return 1

    csv_path, parquet_path, summary_path, failures_path = flush_outputs(
        results,
        failures,
        output_dir,
        args.split,
        include_plots=not args.skip_plots,
    )

    summary = json.loads(summary_path.read_text())

    print(f"Saved metrics csv to {csv_path}")
    print(f"Saved metrics parquet to {parquet_path}")
    print(f"Saved summary json to {summary_path}")
    print(f"Saved failures json to {failures_path}")
    print(
        "Key summary: "
        f"scenarios={summary['num_scenarios']} "
        f"closest_p50={summary['numeric_metrics']['min_distance_obs_m']['p50']:.2f}m "
        f"closest_p95={summary['numeric_metrics']['min_distance_obs_m']['p95']:.2f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
