from typing import List, Dict, Tuple

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pytorch_lightning as pl

from .vectornet import VectorNetTrajPred

from ..metrics import ADE, FDE, minADE, minFDE


class VectorNetLightningModule(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = VectorNetTrajPred(*args, **kwargs)
        self.save_hyperparameters()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        B = batch["target_gt"].shape[0]

        self.log("train/loss_step", losses["loss"], on_step=True, batch_size=B)
        self.log(
            "train/loss", losses["loss"], on_epoch=True, prog_bar=True, batch_size=B
        )

        return {
            **losses,
            "pred": pred,
            "logits": logits,
        }

    def validation_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        B = batch["target_gt"].shape[0]
        metrics = self.calculate_metrics(pred, batch["target_gt"])
        losses.update(metrics)

        self.log("val/loss", losses["loss"], on_epoch=True, prog_bar=True, batch_size=B)
        self.log(
            "val/minADE", metrics["minADE"], on_epoch=True, prog_bar=True, batch_size=B
        )
        self.log(
            "val/minFDE", metrics["minFDE"], on_epoch=True, prog_bar=True, batch_size=B
        )

        return {
            **losses,
            "pred": pred,
            "logits": logits,
        }

    def test_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        B = batch["target_gt"].shape[0]
        metrics = self.calculate_metrics(pred, batch["target_gt"])
        losses.update(metrics)

        self.log(
            "test/loss", losses["loss"], on_epoch=True, prog_bar=True, batch_size=B
        )
        self.log(
            "test/minADE", metrics["minADE"], on_epoch=True, prog_bar=True, batch_size=B
        )
        self.log(
            "test/minFDE", metrics["minFDE"], on_epoch=True, prog_bar=True, batch_size=B
        )

        return {
            **losses,
            "pred": pred,
            "logits": logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        max_epochs = self.trainer.max_epochs
        B = self.cfg.datamodule.batch_size
        len_dataset = self.cfg.datamodule.train_size
        steps_per_epoch = len_dataset // B
        total_steps = max_epochs * steps_per_epoch

        warmup_steps = total_steps * self.cfg.optimizer.warmup_ratio
        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        main = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 关键：按 step 调
                "frequency": 1,
            },
        }

    # cal metrics of minade and minfde
    def calculate_metrics(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate minADE and minFDE for given predictions and ground truth."""
        if pred is None:
            return {}

        if pred.dim() == 3:
            # single modal output -> expand to [B, 1, T, 2]
            pred_for_metrics = pred.unsqueeze(1)
        else:
            pred_for_metrics = pred

        min_ade = minADE(pred_for_metrics, gt).mean()
        min_fde = minFDE(pred_for_metrics, gt).mean()

        return {
            "minADE": min_ade,
            "minFDE": min_fde,
        }

    def create_scenario(self, batch, outputs, index: int = 0):
        def _detach(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().cpu()

        pred = outputs["pred"]  # (b, k, T< 2)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1) if logits is not None else None

        lane_counts: List[int] = batch["lane_counts"]
        agent_counts: List[int] = batch["agent_counts"]

        # as lane and agent are flattened in batch, need to get the start and end index
        lane_start = sum(lane_counts[:index])
        lane_end = lane_start + lane_counts[index]
        agent_start = sum(agent_counts[:index])
        agent_end = agent_start + agent_counts[index]

        scenario_id = batch["scenario_ids"][index] if "scenario_ids" in batch else None

        target_agent_global = batch["target_agent_global_idx"][index].item()
        target_agent_idx = int(target_agent_global - agent_start)

        # TODO: add prediction
        prediction = None
        other_prediction = None
        if pred is not None:
            # [b, t, 2] or [b, k, t, 2] or [b, n, k, t, 2]
            prediction = pred[index]  # [t, 2] or [k, t, 2]
            probabilities = probs[index]  # [k]

        return {
            "lane_points": _detach(batch["lane_points"][lane_start:lane_end]),
            "agent_hist_pos": _detach(batch["agent_history"][agent_start:agent_end]),
            "agent_fut_pos": _detach(batch["agent_future"][agent_start:agent_end]),
            "agent_hist_mask": _detach(
                batch["agent_history_mask"][agent_start:agent_end]
            ),
            "agent_fut_mask": _detach(
                batch["agent_future_mask"][agent_start:agent_end]
            ),
            "agent_last_pos": _detach(batch["agent_last_pos"][agent_start:agent_end]),
            "target_agent_idx": target_agent_idx,
            "preds": prediction,
            "probs": probabilities,
            "scenario_id": scenario_id,
            "k": self.model.k,
            "score_types": None,
            # "log_id": scenario_id,  # TODO: lane plot should in global coords
            "agent_types": batch["agent_types"][agent_start:agent_end],
            "score_types": batch["agent_score_types"][agent_start:agent_end],
        }
