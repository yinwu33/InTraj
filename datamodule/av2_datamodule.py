from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader

from .datasets.av2_dataset import AV2Dataset


class AV2Datamodule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Argoverse 2 motion forecasting."""

    def __init__(
        self,
        data_root: str,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
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
        self.data_root = Path(to_absolute_path(data_root))
        if not self.data_root.exists():
            raise FileNotFoundError(f"{data_root} not found")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.preprocess = preprocess
        self.preprocess_dir = preprocess_dir
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.lane_agent_k = lane_agent_k
        self.lane_radius = lane_radius
        self.agent_radius = agent_radius

        self.train_dataset: AV2Dataset | None = None
        self.val_dataset: AV2Dataset | None = None
        self.test_dataset: AV2Dataset | None = None

    def setup(self, stage: str | None = None):
        common_kwargs = dict(
            data_root=self.data_root,
            preprocess=self.preprocess,
            preprocess_dir=self.preprocess_dir,
            history_steps=self.history_steps,
            future_steps=self.future_steps,
            max_agents=self.max_agents,
            max_lanes=self.max_lanes,
            lane_points=self.lane_points,
            lane_agent_k=self.lane_agent_k,
            lane_radius=self.lane_radius,
            agent_radius=self.agent_radius,
        )

        if stage in ("fit", None):
            self.train_dataset = AV2Dataset(split=self.train_split, **common_kwargs)
            self.val_dataset = AV2Dataset(split=self.val_split, **common_kwargs)

        if stage in ("test", None):
            self.test_dataset = AV2Dataset(split=self.test_split, **common_kwargs)

    def collate_fn(self, batch: List[Dict]):
        lane_offset = 0
        agent_offset = 0

        lane_points = []
        agent_history = []
        lane_lane_edges = []
        agent_agent_edges = []
        lane_agent_edges = []
        target_indices = []
        target_last_pos = []
        target_gt = []
        scenario_ids = []

        for sample in batch:
            lane_points.append(sample["lane_points"])
            agent_history.append(sample["agent_history"])

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
            target_gt.append(sample["target_gt"])
            scenario_ids.append(sample.get("scenario_id", ""))

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
            "target_gt": (
                torch.stack(target_gt)
                if len(target_gt) > 0
                else torch.zeros((0, self.future_steps, 2))
            ),
            "scenario_ids": scenario_ids,
        }
        return batch_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
