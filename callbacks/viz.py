from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib

matplotlib.use("Agg")  # safe headless backend for dataloader forks
import matplotlib.pyplot as plt

from utils.viz import plot_scenario


class TrajectoryVisualizationCallback(pl.Callback):
    """Log a visualization of the first scenario in train and val batches."""

    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self._train_logged = False
        self._val_logged = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._train_logged = False

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._val_logged = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        if self._train_logged or trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        self._log_first_scenario(trainer, pl_module, outputs, batch, stage="train")
        self._train_logged = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._val_logged or trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        self._log_first_scenario(trainer, pl_module, outputs, batch, stage="val")
        self._val_logged = True

    def _log_first_scenario(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        stage: str,
    ) -> None:
        scenario = pl_module.create_scenario(batch, outputs, index=0)
        fig = plot_scenario(**scenario)
        self._log_figure(trainer, fig, tag=f"{stage}/viz")
        plt.close(fig)

    def _log_figure(self, trainer: pl.Trainer, fig, tag: str) -> None:
        logger = trainer.logger
        if logger is None:
            return

        global_step = trainer.global_step
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=global_step)
        elif hasattr(logger, "log_image"):
            logger.log_image(key=tag, images=[fig], step=global_step)
