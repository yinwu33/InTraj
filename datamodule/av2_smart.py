from __future__ import annotations

import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from .datasets.av2 import AV2SMARTDataset
from .datasets.av2.smart_target_builder import SMARTTargetBuilder


class AV2SMARTDatamodule(pl.LightningDataModule):
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
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = os.path.expanduser(os.path.normpath(data_root))
        self.dataset_cfg = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.shuffle = shuffle
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_dataset(self, split: str, use_target_builder: bool) -> AV2SMARTDataset:
        kwargs = OmegaConf.to_container(self.dataset_cfg, resolve=True)
        processed_dir = kwargs.pop("processed_dir", None)
        raw_dir = kwargs.pop("raw_dir", None)
        if raw_dir is None:
            raw_dir = os.path.join(self.data_root, split)
        if processed_dir is None:
            processed_dir = os.path.join("cache", "av2_smart", split)

        transform = None
        if use_target_builder:
            transform = SMARTTargetBuilder(
                num_historical_steps=kwargs["num_historical_steps"],
                num_future_steps=kwargs["num_future_steps"],
            )

        return AV2SMARTDataset(
            root=self.data_root,
            split=split,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            transform=transform,
            **kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(self.train_split, use_target_builder=True)
            self.val_dataset = self._build_dataset(self.val_split, use_target_builder=True)
        if stage in ("test", None):
            self.test_dataset = self._build_dataset(self.test_split, use_target_builder=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

