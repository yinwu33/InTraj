from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from .predictor import SMART as _ReferenceSMART


_AGENT_TYPES = [
    "vehicle",
    "pedestrian",
    "cyclist",
    "background",
]


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    return value


class SMARTLightningModule(_ReferenceSMART):
    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        output_head: bool,
        num_heads: int,
        num_historical_steps: int,
        num_future_steps: int,
        head_dim: int,
        dropout: float,
        num_freq_bands: int,
        warmup_steps: int,
        total_steps: int,
        decoder: dict,
        lr: float,
        agent_token_path: str | None = None,
        map_token_path: str | None = None,
        inference_token: bool = False,
        **kwargs,
    ) -> None:
        decoder_cfg = dict(decoder)
        decoder_cfg.setdefault("num_future_steps", num_future_steps)
        model_config = _to_namespace(
            {
                "dataset": dataset,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "output_head": output_head,
                "num_heads": num_heads,
                "num_historical_steps": num_historical_steps,
                "num_future_steps": num_future_steps,
                "head_dim": head_dim,
                "dropout": dropout,
                "num_freq_bands": num_freq_bands,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "decoder": decoder_cfg,
                "lr": lr,
                "agent_token_path": agent_token_path,
                "map_token_path": map_token_path,
                "inference_token": inference_token,
                **kwargs,
            }
        )
        super().__init__(model_config=model_config)
        self.inference_token = inference_token

    def _get_single_graph(self, batch: HeteroData | Batch, index: int) -> HeteroData:
        if isinstance(batch, Batch):
            return batch.to_data_list()[index]
        return batch

    def create_scenario(self, batch: Any, outputs: Any, index: int = 0):
        data = self._get_single_graph(batch, index)

        lane_points = data["map_save"]["traj_pos"][..., :2].cpu()
        hist_pos = data["agent"]["position"][:, : self.num_historical_steps, :2].cpu()
        hist_vel = data["agent"]["velocity"][:, : self.num_historical_steps, :2].cpu()
        hist_heading = data["agent"]["heading"][:, : self.num_historical_steps].cpu()
        hist_mask = data["agent"]["valid_mask"][:, : self.num_historical_steps].cpu()
        future = data["agent"]["position"][:, self.num_historical_steps :, :2].cpu()
        future_mask = data["agent"]["valid_mask"][:, self.num_historical_steps :].cpu()
        last_pos = data["agent"]["position"][:, self.num_historical_steps - 1, :2].cpu()

        agent_history = torch.cat(
            [
                hist_pos,
                hist_vel,
                hist_heading.sin().unsqueeze(-1),
                hist_heading.cos().unsqueeze(-1),
                hist_mask.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        role = data["agent"]["role"].cpu()
        av_index = int(data["agent"].get("av_index", data["agent"].get("av_idx", -1)))
        category = data["agent"]["category"].cpu()
        focal_candidates = torch.where(role[:, 1])[0]
        scored_candidates = torch.where(category == 3)[0]
        if focal_candidates.numel() > 0:
            target_agent_idx = int(focal_candidates[0].item())
        elif scored_candidates.numel() > 0:
            target_agent_idx = int(scored_candidates[0].item())
        elif av_index >= 0:
            target_agent_idx = av_index
        else:
            target_agent_idx = 0

        preds = future[target_agent_idx : target_agent_idx + 1]
        probs = torch.ones(1)

        score_types = []
        for i in range(data["agent"]["num_nodes"]):
            if i == av_index:
                score_types.append("av")
            elif bool(role[i, 1]):
                score_types.append("focal")
            elif int(category[i]) == 3:
                score_types.append("score")
            else:
                score_types.append("other")

        agent_types = [_AGENT_TYPES[min(int(t), len(_AGENT_TYPES) - 1)] for t in data["agent"]["type"].tolist()]

        return {
            "lane_points": lane_points,
            "agent_history": agent_history,
            "agent_future": future,
            "agent_history_mask": hist_mask,
            "agent_future_mask": future_mask,
            "agent_last_pos": last_pos,
            "target_agent_idx": target_agent_idx,
            "preds": preds,
            "probs": probs,
            "scenario_id": data["scenario_id"],
            "k": 1,
            "score_types": score_types,
            "log_id": None,
            "agent_types": agent_types,
        }

