from typing import List, Dict, Tuple

import torch
import pytorch_lightning as pl

from torch.nn import functional as F

from .vectornet import VectorNetTrajPred

from ..metrics import ADE, FDE, minADE, minFDE


class VectorNetLightningModule(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = VectorNetTrajPred(*args, **kwargs)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(losses, "train", batch_size=batch["target_gt"].shape[0])
        return {
            "loss": losses["loss"],
            "pred": pred,
            "logits": logits,
        }

    def validation_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(losses, "val", batch_size=batch["target_gt"].shape[0])

        if self.model.k == 1:
            # single modal metrics ade, fde
            ade = ADE(pred, batch["target_gt"]).mean()
            fde = FDE(pred, batch["target_gt"]).mean()
            self.log(
                "val/ADE",
                ade,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
            self.log(
                "val/FDE",
                fde,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
        else:
            # multi modal metrics minade, minfde
            min_ade = minADE(pred, batch["target_gt"]).mean()
            min_fde = minFDE(pred, batch["target_gt"]).mean()
            self.log(
                "val/minADE",
                min_ade,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
            self.log(
                "val/minFDE",
                min_fde,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
        return {
            "loss": losses["loss"],
            "pred": pred,
            "logits": logits,
        }

    def test_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(losses, "test", batch_size=batch["target_gt"].shape[0])

        if self.model.k == 1:
            # single modal metrics ade, fde
            ade = ADE(pred, batch["target_gt"]).mean()
            fde = FDE(pred, batch["target_gt"]).mean()
            self.log(
                "test/ADE",
                ade,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
            self.log(
                "test/FDE",
                fde,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
        else:
            # multi modal metrics minade, minfde
            min_ade = minADE(pred, batch["target_gt"]).mean()
            min_fde = minFDE(pred, batch["target_gt"]).mean()
            self.log(
                "test/minADE",
                min_ade,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )
            self.log(
                "test/minFDE",
                min_fde,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["target_gt"].shape[0],
            )

        return {
            "loss": losses["loss"],
            "pred": pred,
            "logits": logits,
        }

    def _log_losses(self, loss_dict, prefix: str, batch_size: int):
        for k, v in loss_dict.items():
            self.log(
                f"{prefix}/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

        prediction = None
        other_prediction = None
        if pred is not None:
            # [b, t, 2] or [b, k, t, 2] or [b, n, k, t, 2]
            prediction = pred[index]  # [t, 2] or [k, t, 2]
            probabilities = probs[index]  # [k]

        return {
            "lane_points": _detach(batch["lane_points"][lane_start:lane_end]),
            "agent_history": _detach(batch["agent_history"][agent_start:agent_end]),
            "agent_future": _detach(batch["agent_future"][agent_start:agent_end]),
            "agent_history_mask": _detach(batch["agent_history_mask"][agent_start:agent_end]),
            "agent_future_mask": _detach(batch["agent_future_mask"][agent_start:agent_end]),
            "agent_last_pos": _detach(batch["agent_last_pos"][agent_start:agent_end]),
            "target_agent_idx": target_agent_idx,
            "preds": prediction,
            "probs": probabilities,
            "scenario_id": scenario_id,
            "k": self.model.k,
            "score_types": None,
            "log_id": scenario_id,
            "agent_types": batch["agent_types"][agent_start:agent_end],
            "score_types": batch["agent_score_types"][agent_start:agent_end],
        }
