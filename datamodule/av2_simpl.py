from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .datasets.av2 import AV2SimplDataset


class AV2SimplDatamodule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Argoverse 2 motion forecasting."""

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
    ):
        super().__init__()

        self.data_root = data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.dataset_cfg = dataset

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        common_kwargs = OmegaConf.to_container(self.dataset_cfg, resolve=True)
        common_kwargs.update({"data_root": self.data_root})

        if stage in ("fit", None):
            self.train_dataset = AV2SimplDataset(
                split=self.train_split, **common_kwargs
            )
            self.val_dataset = AV2SimplDataset(split=self.val_split, **common_kwargs)

        # if stage in ("test", None):
        else:
            self.val_dataset = AV2SimplDataset(split=self.val_split, **common_kwargs)
            self.test_dataset = AV2SimplDataset(split=self.test_split, **common_kwargs)

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

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )
