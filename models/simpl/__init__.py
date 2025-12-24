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

        num_agents = batch["agent_history"].shape[1]
        num_lanes = batch["lane_feats"].shape[1]

        self.log(
            "train/num_agents",
            num_agents,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/num_lanes",
            num_lanes,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "train/num_all",
            num_agents + num_lanes,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self.model.post_process(out)
        pred = post_out["traj_pred"]
        logits = post_out["prob_pred"]

        losses = self.model.loss(out, batch)

        self._log_losses(losses, "train", batch_size=pred.shape[0])
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self.model.post_process(out)
        pred = post_out["traj_pred"]
        logits = post_out["prob_pred"]

        losses = self.model.loss(out, batch)

        self._log_losses(losses, "val", batch_size=pred.shape[0])

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
        res_cls, res_reg, res_aux = self.model(batch)
        out = self.model.post_process((res_cls, res_reg, res_aux))
        pred = out["traj_pred"]
        logits = out["prob_pred"]

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

    def run_forward_postprocess(self, batch: dict) -> torch.Tensor:
        out = self.model(batch)
        out_post = self.model.post_process(out)

        pred = out_post["traj_pred"]
        prob = out_post["prob_pred"]

        return pred, prob

    def create_scenario(self, batch, outputs, index: int = 0):

        def _detach(x):
            if x is None:
                return None
            return x.detach().cpu()

        # * forward pass for predictions (target agent only after post_process)
        with torch.no_grad():
            out = self.model(batch)
        res_cls, res_reg, _ = out

        # gather current sample
        lane_feats = batch["lane_feats"][index]
        lane_mask = batch["lane_masks"][index].bool()
        lane_ctrs = batch["lane_ctrs"][index]
        lane_vecs = batch["lane_vecs"][index]

        agent_history = batch["agent_history"][index]  # (num_agents, 50-2, 14)
        agent_history_mask = batch["agent_history_mask"][index].bool()  # (na, 50-2)
        agent_future_pos = batch["agent_future_pos"][index]
        agent_future_mask = batch["agent_future_mask"][index].bool()

        agent_last_pos = batch["agent_last_pos"][index]

        # gathre predictions
        preds = res_reg[index]  # (num_agents, k, fut_len, 2)
        logits = res_cls[index]  # (num_agents, k)
        probs = torch.softmax(logits, dim=1)  # (num_agents, k)
        focal_agent_idx = 0
        ego_agent_idx = 1

        cum = torch.cumsum(agent_history[:, :, :2], dim=1)  # (N,T, 2)
        agent_first_pos = agent_last_pos - cum[:, -1, :]
        agent_history_pos = cum + agent_first_pos[:, None, :]

        target_agent_idx = 0  # focal agent is first by construction

        # recover lane points for viz
        lane_points = []
        # lane_inputs: (num_lane_segments, num_points_per_lane, 16)
        # 16: node_ctrs(2), node_vecs(2), intersect(1), lane_type(3)
        # cross_left(3), cross_right(3), left_nb(1), right_nb(1)
        for i in range(lane_feats.shape[0]):
            feats = lane_feats[i]  # (N, 16)
            lane_ctr = lane_ctrs[i]  # (2,)
            lane_vec = lane_vecs[i]  # (2,)

            node_ctrs_local = feats[:, 0:2]  # (N, 2)

            vx, vy = lane_vec[0], lane_vec[1]
            R = torch.stack(
                [
                    torch.stack([vx, -vy]),
                    torch.stack([vy, vx]),
                ],
                dim=0,
            )  # (2, 2)

            node_ctrs_scene = node_ctrs_local @ R.T + lane_ctr[None, :]  # (N, 2)
            lane_points.append(node_ctrs_scene)
        lane_points = torch.stack(lane_points, dim=0)  # (num_lanes, num_points, 2)

        return {
            "lane_points": lane_points,
            "agent_history": _detach(agent_history_pos),
            "agent_future": _detach(agent_future_pos),
            "agent_history_mask": _detach(agent_history_mask.bool()),
            "agent_future_mask": _detach(agent_future_mask.bool()),
            "agent_last_pos": _detach(agent_last_pos),
            "target_agent_idx": target_agent_idx,
            "preds": _detach(preds),
            "probs": _detach(probs),
            "scenario_id": batch["scenario_id"][index],
            "k": self.model.k,
        }

