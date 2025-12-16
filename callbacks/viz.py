from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
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
        self._log_first_scenario(trainer, pl_module, batch, stage="train")
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
        self._log_first_scenario(trainer, pl_module, batch, stage="val")
        self._val_logged = True

    def _log_first_scenario(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
        stage: str,
    ) -> None:
        preds = self._run_model(pl_module, batch)
        scenario = self._extract_scenario(batch, preds, scenario_idx=0)
        if scenario is None:
            return

        fig = plot_scenario(
            lane_points=scenario["lane_points"],
            agent_history=scenario["agent_history"],
            target_agent_idx=scenario["target_agent_idx"],
            target_last_pos=scenario["target_last_pos"],
            target_gt=scenario["target_gt"],
            prediction=scenario.get("prediction"),
            other_future=scenario.get("other_future"),
            other_future_mask=scenario.get("other_future_mask"),
            other_prediction=scenario.get("other_prediction"),
            agent_last_pos=scenario.get("agent_last_pos"),
            scenario_id=scenario.get("scenario_id"),
        )
        self._log_figure(trainer, fig, tag=f"{stage}/viz")
        plt.close(fig)

    def _run_model(
        self, pl_module: pl.LightningModule, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            preds = pl_module(batch)
        if was_training:
            pl_module.train()
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds.detach().cpu() if isinstance(preds, torch.Tensor) else None

    def _extract_scenario(
        self,
        batch: Dict[str, Any],
        preds: Optional[torch.Tensor],
        scenario_idx: int,
    ) -> Optional[Dict[str, Any]]:
        lane_counts: List[int] = batch.get("lane_counts", [])
        agent_counts: List[int] = batch.get("agent_counts", [])
        if (
            len(lane_counts) == 0
            or len(agent_counts) == 0
            or scenario_idx >= len(lane_counts)
        ):
            return None

        lane_start = sum(lane_counts[:scenario_idx])
        lane_end = lane_start + lane_counts[scenario_idx]
        agent_start = sum(agent_counts[:scenario_idx])
        agent_end = agent_start + agent_counts[scenario_idx]

        scenario_id = (
            batch.get("scenario_ids", [None] * len(lane_counts))[scenario_idx]
            if "scenario_ids" in batch
            else None
        )

        target_agent_global = batch["target_agent_global_idx"][scenario_idx].item(
        )
        target_agent_idx = int(target_agent_global - agent_start)

        prediction = None
        other_prediction = None
        if preds is not None:
            total_agents = sum(agent_counts)
            if preds.shape[0] == len(agent_counts):
                prediction = preds[scenario_idx]
            elif preds.shape[0] == total_agents:
                scenario_preds = preds[agent_start:agent_end]
                prediction = scenario_preds
                other_prediction = scenario_preds

        return {
            "lane_points": batch["lane_points"][lane_start:lane_end].detach().cpu(),
            "agent_history": batch["agent_history"][agent_start:agent_end]
            .detach()
            .cpu(),
            "target_agent_idx": target_agent_idx,
            "target_last_pos": batch["target_last_pos"][scenario_idx].detach().cpu(),
            "target_gt": batch["target_gt"][scenario_idx].detach().cpu(),
            "prediction": prediction,
            "other_future": batch.get("agent_future", None)[
                agent_start:agent_end
            ].detach().cpu()
            if "agent_future" in batch
            else None,
            "other_future_mask": batch.get("agent_future_mask", None)[
                agent_start:agent_end
            ].detach().cpu()
            if "agent_future_mask" in batch
            else None,
            "agent_last_pos": batch.get("agent_last_pos", None)[
                agent_start:agent_end
            ].detach().cpu()
            if "agent_last_pos" in batch
            else None,
            "other_prediction": other_prediction,
            "scenario_id": scenario_id,
        }

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
