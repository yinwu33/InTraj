from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


def _cast_float64_to_float32(data):
    for key, value in data.items():
        if isinstance(value, dict):
            _cast_float64_to_float32(value)
        elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            data[key] = value.float()
    return data


class SMARTTargetBuilder(BaseTransform):
    def __init__(self, num_historical_steps: int, num_future_steps: int, mode: str = "train") -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode

    def _resolve_ego_index(self, agent: dict) -> int:
        av_idx = int(agent.get("av_idx", -1))
        if av_idx >= 0:
            return av_idx

        role = agent.get("role")
        if role is not None and role.shape[0] > 0:
            ego_candidates = torch.where(role[:, 0])[0]
            if ego_candidates.numel() > 0:
                return int(ego_candidates[0].item())
            focal_candidates = torch.where(role[:, 1])[0]
            if focal_candidates.numel() > 0:
                return int(focal_candidates[0].item())

        scored_candidates = torch.where(agent["category"] > 0)[0]
        if scored_candidates.numel() > 0:
            return int(scored_candidates[0].item())
        return 0

    def _score_agents(self, agent: dict) -> dict:
        agent["category"] = agent["category"].clone()
        av_index = self._resolve_ego_index(agent)
        agent["av_index"] = av_index
        if agent["category"].numel() == 0:
            return agent

        agent["category"].zero_()
        if 0 <= av_index < agent["category"].shape[0]:
            agent["category"][av_index] = 5
            ego_pos = agent["position"][av_index, self.num_historical_steps - 1, :2]
            dist = torch.norm(agent["position"][:, self.num_historical_steps - 1, :2] - ego_pos, dim=-1)
            valid_future = agent["valid_mask"][:, self.num_historical_steps :].any(dim=-1)
            moving_agents = agent["type"] != 3
            scored_mask = (dist < 100.0) & valid_future & moving_agents
            scored_mask[av_index] = False
            agent["category"][scored_mask] = 3
        return agent

    def __call__(self, data) -> HeteroData:
        data = _cast_float64_to_float32(data)
        data["agent"] = self._score_agents(data["agent"])
        return HeteroData(data)

