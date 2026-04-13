from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datamodule import SimplDatamodule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark standardized cache miss/hit performance for the Simpl datamodule "
            "on sequential sample access."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("av2", "waymo"),
        default=("av2", "waymo"),
        help="Dataset sources to benchmark.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Unified split name used by the benchmark.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sequential samples to load per benchmark run.",
    )
    parser.add_argument(
        "--cache-root",
        default="cache/motiondataset_benchmark",
        help=(
            "Benchmark cache root. Defaults to an isolated directory so the benchmark "
            "does not delete the main standardized cache."
        ),
    )
    return parser.parse_args()


def resolve_data_root(source: str) -> str:
    env_key = "AV2_DATA_ROOT" if source == "av2" else "WAYMO_DATA_ROOT"
    value = os.getenv(env_key)
    if not value:
        raise RuntimeError(f"{env_key} is not set")
    return value


def build_dataset_cfg(source: str, cache_root: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "source": source,
            "cache_root": cache_root,
            "builder_kwargs": {},
            "history_steps": 50,
            "future_steps": 60,
            "truncate_steps": 2,
            "radius": 100.0,
            "standardization": {
                "history_steps": 50,
                "future_steps": 60,
                "dt": 0.1,
                "coord_frame": "local",
                "align_heading": True,
                "agents": {
                    "max_agents": 64,
                    "velocity_source": "finite_difference",
                    "heading_source": "prefer_input",
                    "include_size": False,
                },
                "map": {
                    "range_m": 100.0,
                    "precision_m": 0.5,
                    "max_polylines": 256,
                    "points_per_polyline": 11,
                    "crop_shape": "circle",
                    "include_lane_centerlines": True,
                    "include_road_lines": True,
                    "include_road_edges": True,
                    "include_crosswalks": True,
                    "include_speed_bumps": True,
                    "include_driveways": True,
                    "include_drivable_areas": False,
                },
            },
        }
    )


def build_datamodule(
    source: str,
    data_root: str,
    split: str,
    cache_root: str,
) -> SimplDatamodule:
    return SimplDatamodule(
        data_root=data_root,
        dataset=build_dataset_cfg(source, cache_root),
        train_split=split,
        val_split=split,
        test_split=split,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def benchmark_source(
    source: str,
    split: str,
    num_samples: int,
    cache_root: str,
) -> dict[str, float | int | str]:
    data_root = resolve_data_root(source)
    cache_dir = REPO_ROOT / cache_root / source / split
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    cold_dm = build_datamodule(
        source=source,
        data_root=data_root,
        split=split,
        cache_root=cache_root,
    )
    cold_dm.setup("fit")
    cold_dataset = cold_dm.train_dataset
    sample_count = min(num_samples, len(cold_dataset))
    if sample_count == 0:
        raise RuntimeError(f"{source} split={split} dataset is empty")

    start = perf_counter()
    for idx in range(sample_count):
        cold_dataset[idx]
    cold_seconds = perf_counter() - start

    warm_dm = build_datamodule(
        source=source,
        data_root=data_root,
        split=split,
        cache_root=cache_root,
    )
    warm_dm.setup("fit")
    warm_dataset = warm_dm.train_dataset

    start = perf_counter()
    for idx in range(sample_count):
        warm_dataset[idx]
    warm_seconds = perf_counter() - start

    cache_files = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0

    return {
        "source": source,
        "split": split,
        "sample_count": sample_count,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "cold_mean_seconds": cold_seconds / sample_count,
        "warm_mean_seconds": warm_seconds / sample_count,
        "speedup": (cold_seconds / warm_seconds) if warm_seconds > 0 else float("inf"),
        "cache_files": cache_files,
        "cache_dir": str(cache_dir),
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    results = []
    for source in args.sources:
        result = benchmark_source(
            source=source,
            split=args.split,
            num_samples=args.num_samples,
            cache_root=args.cache_root,
        )
        results.append(result)

    print("Cache benchmark results")
    for result in results:
        print(
            f"[{result['source']}] split={result['split']} samples={result['sample_count']} "
            f"cold={result['cold_seconds']:.3f}s ({result['cold_mean_seconds']:.3f}s/sample) "
            f"warm={result['warm_seconds']:.3f}s ({result['warm_mean_seconds']:.3f}s/sample) "
            f"speedup={result['speedup']:.2f}x cache_files={result['cache_files']}"
        )
        print(f"cache_dir={result['cache_dir']}")


if __name__ == "__main__":
    main()
