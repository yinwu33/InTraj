from __future__ import annotations

import argparse
import io
import lmdb
from pathlib import Path
from typing import List
import sys
from rich.progress import track

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datamodule.datasets.av2_dataset import AV2Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess AV2 data into a single LMDB."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing AV2 splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name (e.g., train/val/test/mini_train).",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default=None,
        help="Optional explicit LMDB output path.",
    )
    parser.add_argument(
        "--lmdb_map_size_gb", type=float, default=48.0, help="LMDB map size in GB."
    )
    parser.add_argument("--history_steps", type=int, default=50)
    parser.add_argument("--future_steps", type=int, default=60)
    parser.add_argument("--max_agents", type=int, default=64)
    parser.add_argument("--max_lanes", type=int, default=128)
    parser.add_argument("--lane_points", type=int, default=20)
    parser.add_argument("--lane_agent_k", type=int, default=3)
    parser.add_argument("--lane_radius", type=float, default=150.0)
    parser.add_argument("--agent_radius", type=float, default=30.0)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing LMDB file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    lmdb_path = (
        Path(args.lmdb_path)
        if args.lmdb_path
        else data_root / "cache" / f"{args.split}.lmdb"
    )
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    if lmdb_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"LMDB already exists at {lmdb_path}. Use --overwrite to replace."
            )
        lmdb_path.unlink()

    ds = AV2Dataset(
        data_root=data_root,
        split=args.split,
        use_lmdb=False,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        max_agents=args.max_agents,
        max_lanes=args.max_lanes,
        lane_points=args.lane_points,
        lane_agent_k=args.lane_agent_k,
        lane_radius=args.lane_radius,
        agent_radius=args.agent_radius,
    )

    env = lmdb.open(
        str(lmdb_path),
        map_size=int(args.lmdb_map_size_gb * (1024**3)),
        subdir=False,
        lock=True,
        readonly=False,
        meminit=False,
        map_async=False,
    )

    keys: List[str] = []
    for idx, log_dir in track(
        enumerate(ds.log_dirs),
        total=len(ds.log_dirs),
        description="Preprocessing",
    ):
        sample = ds._process_log_dir(log_dir)
        key = log_dir.name.encode("utf-8")
        with io.BytesIO() as bio:
            torch.save(sample, bio)
            buf = bio.getvalue()
        with env.begin(write=True) as txn:
            txn.put(key, buf)
        keys.append(log_dir.name)
        # if (idx + 1) % 10 == 0 or idx == len(ds.log_dirs) - 1:
        #     print(f"Processed {idx + 1}/{len(ds.log_dirs)} scenarios into {lmdb_path}")

    # store keys for fast lookup
    with io.BytesIO() as bio:
        torch.save(keys, bio)
        buf = bio.getvalue()
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", buf)

    env.sync()
    env.close()
    print(f"Finished preprocessing. LMDB stored at: {lmdb_path}")


if __name__ == "__main__":
    main()
