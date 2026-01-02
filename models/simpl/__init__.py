from typing import Optional

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pytorch_lightning as pl

from .simpl import Simpl

from ..metrics import minADE, minFDE


class SimplLightningModule(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        self.model = Simpl(*args, **kwargs)
        self.k = self.model.pred_net.k
        
        self.save_hyperparameters()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)

        B = len(post_out[0])

        self.log("train/loss_step", losses["loss"], on_step=True, batch_size=B)
        self.log(
            "train/loss", losses["loss"], on_epoch=True, prog_bar=True, batch_size=B
        )
        return {
            **losses,
            "pred": post_out[1],
            "logits": post_out[0],
        }

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)

        B = len(post_out[0])

        metrics = self.calculate_metrics(post_out[1], batch)
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
            "pred": post_out[1],
            "logits": post_out[0],
        }

    def test_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)

        B = len(post_out[0])

        metrics = self.calculate_metrics(post_out[1], batch)
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
            "pred": post_out[1],
            "logits": post_out[0],
        }

    def _post_process(self, out: tuple, batch: dict) -> dict:
        res_cls, res_reg, res_aux = out
        agent_last_pos_global = batch["agent_last_pos"]
        agent_last_rot_global = batch["agent_last_rot"]

        agent_history_mask = batch["agent_history_mask"].bool()

        B = len(res_reg)

        res_reg_global = []
        for i in range(B):
            mask = agent_history_mask[i]
            valid_agents = mask.any(dim=1)  # (num_agents,)
            R = agent_last_rot_global[i][valid_agents]  # (num_valid, 2, 2)
            t = agent_last_pos_global[i][valid_agents]  # (num_valid,
            res_reg_global.append(
                self._agent_frame_to_global(res_reg[i], R, t, k_dim=1)
            )

        return res_cls, res_reg_global, res_aux

    def calculate_metrics(self, pred: list, batch: torch.Tensor) -> dict:
        """Calculate minADE and minFDE for given predictions and ground truth.
        pred: list of [(num_agents, k, T, 2)]
        gt: (B, num_agents, T, 2)
        """
        B = len(pred)

        agent_future_pos = batch["agent_future_pos"]
        agent_history_mask = batch["agent_history_mask"].bool()

        min_ade_list = []
        min_fde_list = []

        for i in range(B):
            i_pred = pred[i]  # (num_agents, k, T, 2)
            i_gt = agent_future_pos[i]  # (num_agents, T, 2)

            mask = agent_history_mask[i]
            valid_agents = mask.any(dim=1)  # (num_agents,)
            i_gt = i_gt[valid_agents]  # (num_valid, T, 2)

            min_ade = minADE(i_pred, i_gt).mean()
            min_fde = minFDE(i_pred, i_gt).mean()
            min_ade_list.append(min_ade)
            min_fde_list.append(min_fde)

        min_ade = torch.stack(min_ade_list).mean()
        min_fde = torch.stack(min_fde_list).mean()

        return {
            "minADE": min_ade,
            "minFDE": min_fde,
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

    def run_forward_postprocess(self, batch: dict) -> torch.Tensor:
        out = self.model(batch)
        out_post = self.model.post_process(out)

        pred = out_post["traj_pred"]
        prob = out_post["prob_pred"]

        return pred, prob

    def _agent_frame_to_global(self, pts, R, t, k_dim=None):
        # pts: (..., N, T, 2)
        # R:   (..., N, 2, 2)
        # t:   (..., N, 2)
        if k_dim is None:
            pts_global = R @ pts.transpose(-1, -2)  # 结果: (..., N, 2, T)
            pts_global = pts_global.transpose(-1, -2) + t.unsqueeze(-2)

            return pts_global
        else:
            # e.g., k_dim = 1, means pts: (N, k, T, 2)
            # R:   (N, 2, 2)
            # t:   (N, 2)
            Rk = R.unsqueeze(k_dim)  # (..., N, 1, 2, 2)  (假设 k_dim 指向 K 那一维)
            tk = t.unsqueeze(k_dim).unsqueeze(-2)  # (..., N, 1, 1, 2)

            # (..., N, K, 2, 2) @ (..., N, K, 2, T) -> (..., N, K, 2, T)
            pts_global = Rk @ pts.transpose(-1, -2)
            pts_global = pts_global.transpose(-1, -2) + tk  # (..., N, K, T, 2)
            return pts_global

    def _get_agent_types(self, batch, index: int = 0):
        # ObjectType.VEHICLE: 0,
        # ObjectType.PEDESTRIAN: 1,
        # ObjectType.MOTORCYCLIST: 2,
        # ObjectType.CYCLIST: 3,
        # ObjectType.BUS: 4,
        # ObjectType.UNKNOWN: 5,
        # agent_history is 14 with 2+2+2+7+1
        # where 7 is one-hot encoding of object type {vehicle, pedestrian, motorcyclist, cyclist, bus, unknown, default}
        agent_history = batch["agent_history"][index]  # (num_agents, 50-2, 14)
        agent_history_mask = batch["agent_history_mask"][index].bool()  # (na, 50-2)

        valid_agent_history = agent_history[agent_history_mask.any(-1)]
        agent_types = []
        for agent in valid_agent_history:
            obj_type_onehot = agent[:, 7:14].sum(dim=0)  # (7,)
            obj_type_idx = torch.argmax(obj_type_onehot).item()
            if obj_type_idx == 0:
                agent_types.append("vehicle")
            elif obj_type_idx == 1:
                agent_types.append("pedestrian")
            elif obj_type_idx == 2:
                agent_types.append("motorcyclist")
            elif obj_type_idx == 3:
                agent_types.append("cyclist")
            elif obj_type_idx == 4:
                agent_types.append("bus")
            else:
                agent_types.append("unknown")
        return agent_types

    def create_scenario(self, batch, outputs, index: int = 0):

        def _detach(x):
            if x is None:
                return None
            return x.detach().cpu()

        # * forward pass for predictions (target agent only after post_process)
        with torch.no_grad():
            out = self.model(batch)
            post_out = self._post_process(out, batch)
        logits = post_out[0][index]  # traj logits for all agents
        preds = post_out[1][index]  # traj preds for all agents
        probs = torch.softmax(logits, dim=0)

        # gather current sample
        lane_feats = batch["lane_feats"][index]
        node_ctrs = lane_feats[:, :, :2]
        node_vecs = lane_feats[:, :, 2:4]

        node_pts = node_ctrs - node_vecs * 0.5  # (num_nodes, 2)
        # add one more pts
        node_pts_shifted = node_ctrs + node_vecs * 0.5
        # node_pts = torch.cat([node_pts, node_pts_shifted[-1, :].unsqueeze(0)], dim=0)  # (num_nodes+1, 2)

        lane_mask = batch["lane_masks"][index]
        lane_anchor_points_global = batch["lane_ctrs"][index]
        lane_anchor_vecs_globals = batch["lane_vecs"][index]

        agent_history = batch["agent_history"][index]  # (num_agents, 50-2, 14)
        agent_history_mask = batch["agent_history_mask"][index].bool()  # (na, 50-2)
        agent_future_pos_global = batch["agent_future_pos"][index]
        agent_future_mask = batch["agent_future_mask"][index].bool()

        agent_last_pos_global = batch["agent_last_pos"][index]
        agent_last_rot_global = batch["agent_last_rot"][index]
        agent_types = self._get_agent_types(batch, index)

        # gathre predictions
        focal_agent_idx = 0
        ego_agent_idx = 1

        cum = torch.cumsum(agent_history[:, :, :2], dim=1)  # (N,T, 2)
        agent_first_pos = (
            torch.zeros([agent_history.shape[0], 2], device=cum.device) - cum[:, -1, :]
        )
        agent_history_pos_local = cum + agent_first_pos[:, None, :]  # (N,T,2)

        # now the agent_history_pos is still in local agent frame
        # need to transform back to scene frame
        # agent_future_pos_global = []
        # preds_global = []

        agent_history_pos_global = self._agent_frame_to_global(
            agent_history_pos_local,
            agent_last_rot_global,
            agent_last_pos_global,
        )

        # preds_global = torch.stack(preds_global, dim=0)

        target_agent_idx = 0  # focal agent is first by construction

        target_agent_last_pos = agent_last_pos_global[target_agent_idx]  #
        target_agent_last_rot = agent_last_rot_global[target_agent_idx]  #

        lane_anchor_rot_global = torch.zeros(
            (lane_anchor_vecs_globals.shape[0], 2, 2),
            device=lane_anchor_vecs_globals.device,
        )
        lane_anchor_rot_global[:, 0, 0] = lane_anchor_vecs_globals[:, 0]
        lane_anchor_rot_global[:, 0, 1] = -lane_anchor_vecs_globals[:, 1]
        lane_anchor_rot_global[:, 1, 0] = lane_anchor_vecs_globals[:, 1]
        lane_anchor_rot_global[:, 1, 1] = lane_anchor_vecs_globals[:, 0]

        lane_pts_global = lane_anchor_rot_global @ node_pts.transpose(1, 2)
        lane_pts_global = (
            lane_pts_global.transpose(1, 2) + lane_anchor_points_global[:, None, :]
        )

        return {
            "lane_points": lane_pts_global,
            "agent_hist_pos": _detach(agent_history_pos_global),
            "agent_fut_pos": _detach(agent_future_pos_global),
            "agent_hist_mask": _detach(agent_history_mask.bool()),
            "agent_fut_mask": _detach(agent_future_mask.bool()),
            "agent_last_pos": _detach(agent_last_pos_global),
            "target_agent_idx": target_agent_idx,
            "preds": _detach(preds),
            "probs": _detach(probs),
            "scenario_id": batch["scenario_id"][index],
            "k": self.model.k,
            "score_types": batch["agent_score_types"][index],
            "log_id": batch["scenario_id"][index],
            "agent_types": agent_types,
        }
