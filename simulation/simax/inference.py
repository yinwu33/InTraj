"""SIMPL inference engine for trajectory prediction.

Loads a :class:`Simpl` model (optionally from a checkpoint) and exposes a
simple :meth:`predict` method that returns future trajectory positions for
the requested target vehicles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from models.simpl.simpl import Simpl


# Default hyper-parameters (matches configs/model/simpl.yaml).
# TODO: should consider use hydra with default configs
_DEFAULT_HPARAMS = {
    "actor_net_cfg": {"input_dim": 14, "hidden_dim": 128, "num_fpn_scale": 4},
    "lane_net_cfg": {"input_dim": 16, "hidden_dim": 128, "dropout": 0.1},
    "fusion_net_cfg": {
        "actor_emb_dim": 128,
        "lane_emb_dim": 128,
        "rpe_input_dim": 5,
        "rpe_emb_dim": 128,
        "hidden_dim": 128,
        "dropout": 0.1,
        "num_scene_heads": 8,
        "num_scene_layers": 4,
        "update_edge": True,
    },
    "mlp_decoder_cfg": {
        "hidden_dim": 128,
        "global_pred_lane": 60,
        "k": 6,
        "param_out": "bezier",
        "param_order": 7,
    },
    "loss_cfg": {
        "global_pred_lane": 60,
        "k": 6,
        "reg_coef": 0.9,
        "cls_coef": 0.1,
        "mgn": 0.2,
        "cls_thres": 2.0,
        "cls_ignore": 0.2,
        "yaw_loss": False,
    },
}


class SimplInferenceEngine:
    """Wraps :class:`Simpl` for single-forward-pass inference.

    Parameters
    ----------
    checkpoint_path
        Path to a PyTorch Lightning ``.ckpt`` or raw ``.pt`` file.
        If *None* the model runs with random weights.
    device
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Try to extract hyper-parameters from the checkpoint itself.
        hparams = dict(_DEFAULT_HPARAMS)
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "hyper_parameters" in raw:
                    hp = raw["hyper_parameters"]
                    for key in hparams:
                        if key in hp:
                            hparams[key] = hp[key]

        self.k = hparams["mlp_decoder_cfg"]["k"]
        self.future_steps = hparams["mlp_decoder_cfg"]["global_pred_lane"]

        self.model = Simpl(
            actor_net_cfg=OmegaConf.create(hparams["actor_net_cfg"]),
            lane_net_cfg=OmegaConf.create(hparams["lane_net_cfg"]),
            fusion_net_cfg=OmegaConf.create(hparams["fusion_net_cfg"]),
            mlp_decoder_cfg=OmegaConf.create(hparams["mlp_decoder_cfg"]),
            loss_cfg=OmegaConf.create(hparams["loss_cfg"]),
        )

        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                print(
                    f"[inference] WARNING: checkpoint {checkpoint_path} not found — "
                    "using untrained weights."
                )
            else:
                self._load_checkpoint(ckpt_path)

        self.model.to(self.device)
        self.model.eval()
        print(
            f"[inference] SIMPL ready  (k={self.k}, future_steps={self.future_steps}, "
            f"device={self.device}, weights={'random' if checkpoint_path is None else checkpoint_path})"
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict) -> np.ndarray:
        """Run a forward pass and return the best-mode trajectory.

        Parameters
        ----------
        batch
            Dict produced by :meth:`SimaxSimplConverter.build_batch`.
            Must contain ``agent_last_pos`` and ``agent_last_rot`` for the
            coordinate back-transform.

        Returns
        -------
        np.ndarray
            Predicted positions with shape ``[N_targets, future_steps, 2]``
            in the **global** coordinate frame (same as simax positions).
        """
        # Move tensors to device (skip lists / metadata).
        dev_batch: dict = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                dev_batch[k] = v.to(self.device)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                dev_batch[k] = [t.to(self.device) for t in v]
            else:
                dev_batch[k] = v

        out = self.model(dev_batch)
        post = self.model.post_process(out)
        traj_pred = post["traj_pred"]  # [B, k, 60, 2]  agent-local
        prob_pred = post["prob_pred"]  # [B, k]

        # Select best mode per sample.
        best_k = prob_pred.argmax(dim=-1)  # [B]
        B = traj_pred.shape[0]
        best_traj = traj_pred[torch.arange(B, device=traj_pred.device), best_k]
        # best_traj: [B, 60, 2]  (agent-local frame)

        # Transform back to global coordinates.
        # target agent is at index 0 in each batch item.
        agent_last_pos = dev_batch["agent_last_pos"][:, 0, :]    # [B, 2]
        agent_last_rot = dev_batch["agent_last_rot"][:, 0, :, :]  # [B, 2, 2]

        # Inverse rotation: R^T (since R is orthogonal).
        rot_inv = agent_last_rot.transpose(-1, -2)  # [B, 2, 2]
        global_traj = torch.einsum("btd,bde->bte", best_traj, rot_inv) + agent_last_pos[:, None, :]
        # global_traj: [B, 60, 2]

        return global_traj.cpu().numpy()

    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        """Load weights from a Lightning ``.ckpt`` or raw ``.pt`` file."""
        raw = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" in raw:
            sd = {
                k.removeprefix("model."): v
                for k, v in raw["state_dict"].items()
            }
        else:
            sd = raw

        self.model.load_state_dict(sd, strict=False)
        print(f"[inference] Loaded weights from {path}")
