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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not isinstance(batch, dict) or "motion_samples" not in batch:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        batch_on_device = dict(batch)
        motion_samples = batch_on_device.pop("motion_samples")
        batch_on_device = super().transfer_batch_to_device(
            batch_on_device, device, dataloader_idx
        )
        batch_on_device["motion_samples"] = motion_samples
        return batch_on_device

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
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self._post_process(out, batch)
        losses = self.model.loss(post_out, batch)
        B = len(post_out[0])
        self.log(
            "test/loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        target_gt = batch.get("target_gt")
        if target_gt is not None:
            pred = post_out[1]  # res_reg_global
            if self.model.k == 1:
                # single modal metrics ade, fde
                ade = ADE(pred, target_gt).mean()
                fde = FDE(pred, target_gt).mean()
                self.log("test/ADE", ade, prog_bar=True, on_epoch=True, batch_size=B)
                self.log("test/FDE", fde, prog_bar=True, on_epoch=True, batch_size=B)
            else:
                # multi modal metrics minade, minfde
                min_ade = minADE(pred, target_gt).mean()
                min_fde = minFDE(pred, target_gt).mean()
                self.log("test/minADE", min_ade, prog_bar=True, on_epoch=True, batch_size=B)
                self.log("test/minFDE", min_fde, prog_bar=True, on_epoch=True, batch_size=B)

        return {
            "loss": losses["loss"],
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
        
        
    def _get_agent_types(self, batch, index: int = 0):
        simpl_agent_types = (
            "vehicle",
            "pedestrian",
            "cyclist",
            "motorcyclist",
            "bus",
            "static",
            "unknown",
        )

        agent_type_indices = batch["agent_type"][index]
        agent_history_mask = batch["agent_history_mask"][index].bool()

        agent_types = []
        for agent_idx in range(agent_history_mask.shape[0]):
            if not agent_history_mask[agent_idx].any():
                agent_types.append("unknown")
                continue

            obj_type_idx = int(agent_type_indices[agent_idx].item())
            if not 0 <= obj_type_idx < len(simpl_agent_types):
                raise ValueError(
                    f"Unexpected SIMPL agent type index {obj_type_idx} at sample {index}, agent {agent_idx}"
                )
            agent_types.append(simpl_agent_types[obj_type_idx])
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
        probs = post_out[0][index]  # traj probs for all agents
        preds = post_out[1][index]  # traj preds for all agents

        # keep per-agent modal probabilities normalized on mode dimension
        prob_sums = probs.sum(dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3, rtol=1e-3):
            probs = torch.softmax(probs, dim=-1)

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
            "agent_history": _detach(agent_history_pos_global),
            "agent_future": _detach(agent_future_pos_global),
            "agent_history_mask": _detach(agent_history_mask.bool()),
            "agent_future_mask": _detach(agent_future_mask.bool()),
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
