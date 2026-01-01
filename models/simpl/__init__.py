from typing import Optional

import torch
import pytorch_lightning as pl

from .simpl import Simpl

from ..metrics import ADE, FDE, minADE, minFDE


class SimplLightningModule(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Simpl(*args, **kwargs)
        self.k = self.model.pred_net.k

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        # post_out = self.model.post_process(out)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)

        B = len(post_out[0])

        self.log(
            "train/loss_step",
            losses["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=B,
        )
        self.log(
            "train/loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        # post_out = self.model.post_process(out)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)
        B = len(post_out[0])
        self.log(
            "val/loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        # self._log_min_metrics(post_out, batch, prefix="val")

        # if self.model.k == 1:
        #     # single modal metrics ade, fde
        #     ade = ADE(pred, batch["target_gt"]).mean()
        #     fde = FDE(pred, batch["target_gt"]).mean()
        #     self.log("val/ADE", ade, prog_bar=True, on_epoch=True,
        #              batch_size=batch["target_gt"].shape[0])
        #     self.log("val/FDE", fde, prog_bar=True, on_epoch=True,
        #              batch_size=batch["target_gt"].shape[0])
        # else:
        #     # multi modal metrics minade, minfde
        #     min_ade = minADE(pred, batch["target_gt"]).mean()
        #     min_fde = minFDE(pred, batch["target_gt"]).mean()
        #     self.log("val/minADE", min_ade, prog_bar=True,
        #              on_epoch=True, batch_size=batch["target_gt"].shape[0])
        #     self.log("val/minFDE", min_fde, prog_bar=True,
        #              on_epoch=True, batch_size=batch["target_gt"].shape[0])

    def test_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)
        
        metrics = self.calculate_metrics(post_out[1], batch["target_gt"])
        losses.update(metrics)
        
        self.log(
            "test/loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(post_out[0]),
        )

        self._log_min_metrics(post_out, batch, prefix="test")
        return losses["loss"]

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

    def calculate_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

    # def _log_min_metrics(self, post_out: tuple, batch: dict, prefix: str):
    #     """Compute and log minADE/FDE for the target agent."""
    #     target_preds = self._get_target_preds(post_out)
    #     if target_preds is None:
    #         return

    #     if target_preds.dim() == 3:
    #         target_preds = target_preds.unsqueeze(1)

    #     target_gt = batch["agent_future_pos"][:, 0]

    #     min_ade = minADE(target_preds, target_gt).mean()
    #     min_fde = minFDE(target_preds, target_gt).mean()

    #     batch_size = target_gt.shape[0]
    #     self.log(
    #         f"{prefix}/minADE",
    #         min_ade,
    #         prog_bar=True,
    #         on_epoch=True,
    #         batch_size=batch_size,
    #     )
    #     self.log(
    #         f"{prefix}/minFDE",
    #         min_fde,
    #         prog_bar=True,
    #         on_epoch=True,
    #         batch_size=batch_size,
    #     )

    # def _get_target_preds(self, post_out: tuple) -> Optional[torch.Tensor]:
    #     res_reg_global = post_out[1]
    #     if res_reg_global is None or len(res_reg_global) == 0:
    #         return None

    #     target_preds = []
    #     for sample_preds in res_reg_global:
    #         if sample_preds.shape[0] == 0:
    #             continue
    #         target_preds.append(sample_preds[0])

    #     if not target_preds:
    #         return None

    #     return torch.stack(target_preds, dim=0)

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
