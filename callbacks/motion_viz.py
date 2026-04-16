from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from datasets import MotionScenario
from utils.viz_motion import plot_scenario


class MotionVizCallback(pl.Callback):
    """Log MotionScenario-based visualizations when batches carry standardized samples."""

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
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if self._train_logged or trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not self._has_motion_samples(batch):
            return
        self._log_first_scenario(trainer, pl_module, batch, stage="train")
        self._train_logged = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._val_logged or trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not self._has_motion_samples(batch):
            return
        self._log_first_scenario(trainer, pl_module, batch, stage="val")
        self._val_logged = True

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx > 0:
            return
        if not self._has_motion_samples(batch):
            return
        self._log_first_scenario(trainer, pl_module, batch, stage="test")
        plt.close("all")

    def _has_motion_samples(self, batch: Any) -> bool:
        return isinstance(batch, dict) and bool(batch.get("motion_samples"))

    def _get_motion_sample(self, batch: dict[str, Any], index: int) -> MotionScenario | None:
        samples = batch.get("motion_samples")
        if not isinstance(samples, list) or not (0 <= index < len(samples)):
            return None
        sample = samples[index]
        return sample if isinstance(sample, MotionScenario) else None

    def _infer_target_agent_idx(
        self, batch: dict[str, Any], index: int
    ) -> int | None:
        target_mask = batch.get("target_mask")
        if target_mask is None:
            return None

        target_mask_i = target_mask[index]
        if hasattr(target_mask_i, "detach"):
            target_mask_i = target_mask_i.detach()
        if hasattr(target_mask_i, "cpu"):
            target_mask_i = target_mask_i.cpu()

        target_indices = torch.where(torch.as_tensor(target_mask_i).bool())[0]
        if target_indices.numel() == 0:
            return None
        return int(target_indices[0].item())

    def _compute_predictions(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
        index: int,
    ) -> tuple[Any, Any]:
        if not hasattr(pl_module, "model") or not hasattr(pl_module, "_post_process"):
            return None, None

        with torch.no_grad():
            out = pl_module.model(batch)
            probs, preds, _ = pl_module._post_process(out, batch)

        probs_i = probs[index]
        preds_i = preds[index]
        if hasattr(probs_i, "sum"):
            prob_sums = probs_i.sum(dim=-1)
            if not torch.allclose(
                prob_sums,
                torch.ones_like(prob_sums),
                atol=1e-3,
                rtol=1e-3,
            ):
                probs_i = torch.softmax(probs_i, dim=-1)
        return preds_i.detach().cpu(), probs_i.detach().cpu()

    def _log_first_scenario(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
        stage: str,
    ) -> None:
        sample = self._get_motion_sample(batch, index=0)
        if sample is None:
            return

        preds, probs = self._compute_predictions(pl_module, batch, index=0)
        target_agent_idx = self._infer_target_agent_idx(batch, index=0)
        k = int(getattr(getattr(pl_module, "model", None), "k", getattr(pl_module, "k", 1)))

        fig = plot_scenario(
            sample=sample,
            preds=preds,
            probs=probs,
            k=k,
            target_agent_idx=target_agent_idx,
        )
        self._log_figure(trainer, fig, tag=f"{stage}/motion_viz")
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
