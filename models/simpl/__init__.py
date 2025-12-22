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

        num_agents = batch["agent_feats"].shape[1]
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
        """
        batch: dict
        agent_feats: (b, n_agent_max, dim, t)
        agent_masks: (b, n_agent_max)
        lane_feats: (b, n_lane_max, dim, t)
        lane_masks: (b, n_lane_max)
        "TRAJS_FUT": (b, n_agent_max, future_len, 2)
        "PAD_FUT": (b, n_agent_max, future_len)
        
        return dict:
        lane_points: tensor, (num_lanes, num_points, 2)
        agent_history: tensor, (num_agents, hist_len, feat_dim)
        target_agent_idx: int
        target_last_pos: tensor, (2,)
        target_gt: tensor, (future_len, 2)
        prediction: tensor, (k, future_len, 2)
        other_future: tensor, (num_agents, future_len, 2) or None
        other_future_mask: tensor, (num_agents, future_len) or None
        other_prediction: tensor, (num_agents, k, future_len, 2) or None
        agent_last_pos: tensor, (num_agents, 2) or None
        """
        def _detach(x):
            if x is None:
                return None
            return x.detach().cpu()

        # * forward pass for predictions (target agent only after post_process)
        with torch.no_grad():
            out = self.model(batch)
        res_cls, res_reg, _ = out

        # gather current sample
        agent_inputs = batch["agent_feats"][index]  # (num_agents, feat_dim, hist_len)
        agent_mask = batch["agent_masks"][index].bool()
        lane_inputs = batch["lane_feats"][index]
        lane_mask = batch["lane_masks"][index].bool()

        # keep only valid agents/lanes
        agent_inputs = agent_inputs[agent_mask]
        lane_inputs = lane_inputs[lane_mask]

        # future GT and masks
        future = batch["TRAJS_POS_FUT"][index]
        future_mask = batch["PAD_FUT"][index].bool()
        train_mask = batch.get("TRAIN_MASK", None)
        if train_mask is not None:
            train_mask = train_mask[index].bool()

        # predictions (model outputs are already softmaxed)
        sample_cls = res_cls[index] if res_cls is not None else None
        sample_reg = res_reg[index] if res_reg is not None else None
        prediction = sample_reg[0] if sample_reg is not None else None
        probabilities = sample_cls[0] if sample_cls is not None else None
        other_prediction = sample_reg

        # align auxiliary tensors to number of agents kept
        num_agents = agent_inputs.shape[0]
        future = future[:num_agents]
        future_mask = future_mask[:num_agents]
        if train_mask is not None:
            train_mask = train_mask[:num_agents]
        if other_prediction is not None:
            other_prediction = other_prediction[:num_agents]

        # reconstruct agent history in VectorNet viz format [x, y, vx, vy, sin, cos, mask]
        hist_len = agent_inputs.shape[-1]
        agent_history = agent_inputs.new_zeros((num_agents, hist_len, 7))
        agent_last_pos = agent_inputs.new_zeros((num_agents, 2))
        dt = 0.1  # AV2 timestep

        for i in range(num_agents):
            feats = agent_inputs[i]  # (feat_dim, hist_len)
            disp = feats[0:2]  # (2, hist_len), pos_t - pos_{t-1}
            vel = feats[4:6].T  # (hist_len, 2)
            yaw_cos = feats[2]
            yaw_sin = feats[3]
            obs_mask = feats[-1] > 0.5

            fut_i = future[i] if future is not None else None
            fut_mask_i = future_mask[i] if future_mask is not None else None
            if fut_i is not None:
                valid_future = fut_mask_i if fut_mask_i.dtype == torch.bool else fut_mask_i > 0.5
                if valid_future.any():
                    first_valid_idx = int(torch.nonzero(valid_future, as_tuple=False)[0])
                else:
                    first_valid_idx = 0
                anchor_pos = fut_i[first_valid_idx]
            else:
                anchor_pos = disp.new_zeros(2)

            # refine anchor with last observed velocity to approximate last obs position
            anchor_pos = anchor_pos - feats[4:6, -1] * dt

            pos_seq = feats.new_zeros((hist_len, 2))
            pos_seq[-1] = anchor_pos
            for t in range(hist_len - 1, 0, -1):
                pos_seq[t - 1] = pos_seq[t] - disp[:, t]

            agent_history[i, :, 0:2] = pos_seq
            agent_history[i, :, 2:4] = vel
            agent_history[i, :, 4] = yaw_sin
            agent_history[i, :, 5] = yaw_cos
            agent_history[i, :, 6] = obs_mask.float()
            agent_last_pos[i] = pos_seq[-1]

        target_agent_idx = 0  # focal agent is first by construction

        return {
            "lane_points": _detach(lane_inputs[..., :2]),
            "agent_history": _detach(agent_history),
            "target_agent_idx": target_agent_idx,
            "target_last_pos": _detach(agent_history[target_agent_idx, -1, :2]),
            "target_gt": _detach(future[target_agent_idx]),
            "prediction": _detach(prediction),
            "probabilities": _detach(probabilities),
            "other_future": _detach(future),
            "other_future_mask": _detach(future_mask),
            "agent_last_pos": _detach(agent_last_pos),
            "other_prediction": _detach(other_prediction),
            "scenario_id": None,
        }
