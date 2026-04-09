from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from .predictor import QCNet as _ReferenceQCNet


_AGENT_TYPES = [
    "vehicle",
    "pedestrian",
    "motorcyclist",
    "cyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
    "unknown",
]


class QCNetLightningModule(_ReferenceQCNet):
    def _get_single_graph(self, batch: HeteroData | Batch, index: int) -> HeteroData:
        if isinstance(batch, Batch):
            return batch.to_data_list()[index]
        return batch

    def _predict_single_graph(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        data = data.to(self.device)
        pred = self(data)
        traj_refine = pred["loc_refine_pos"][..., : self.output_dim]
        theta = data["agent"]["heading"][:, self.num_historical_steps - 1]
        origin = data["agent"]["position"][:, self.num_historical_steps - 1, :2]
        cos, sin = theta.cos(), theta.sin()
        rot = torch.zeros(theta.size(0), 2, 2, device=self.device)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        traj_global = torch.matmul(traj_refine, rot.unsqueeze(1)) + origin[:, None, None, :]
        return traj_global.detach().cpu(), F.softmax(pred["pi"], dim=-1).detach().cpu()

    def create_scenario(self, batch: Any, outputs: Any, index: int = 0):
        data = self._get_single_graph(batch, index)
        with torch.no_grad():
            preds, probs = self._predict_single_graph(data)

        point_to_polygon = data["map_point", "to", "map_polygon"]["edge_index"].cpu()
        point_pos = data["map_point"]["position"][:, :2].cpu()
        num_polygons = data["map_polygon"]["num_nodes"]
        lane_polylines = []
        max_len = 0
        for poly_idx in range(num_polygons):
            mask = point_to_polygon[1] == poly_idx
            poly_points = point_pos[point_to_polygon[0][mask]]
            if poly_points.numel() == 0:
                poly_points = data["map_polygon"]["position"][poly_idx : poly_idx + 1, :2].cpu()
            lane_polylines.append(poly_points)
            max_len = max(max_len, poly_points.size(0))
        padded_lanes = []
        for poly_points in lane_polylines:
            if poly_points.size(0) < max_len:
                pad = poly_points[-1:].repeat(max_len - poly_points.size(0), 1)
                poly_points = torch.cat([poly_points, pad], dim=0)
            padded_lanes.append(poly_points)
        lane_points = torch.stack(padded_lanes, dim=0) if padded_lanes else torch.zeros((0, 2, 2))

        hist_pos = data["agent"]["position"][:, : self.num_historical_steps, :2].cpu()
        hist_vel = data["agent"]["velocity"][:, : self.num_historical_steps, :2].cpu()
        hist_heading = data["agent"]["heading"][:, : self.num_historical_steps].cpu()
        observed = data["agent"]["valid_mask"][:, : self.num_historical_steps].cpu().float()
        agent_history = torch.cat(
            [
                hist_pos,
                hist_vel,
                hist_heading.sin().unsqueeze(-1),
                hist_heading.cos().unsqueeze(-1),
                observed.unsqueeze(-1),
            ],
            dim=-1,
        )
        agent_future = data["agent"]["position"][
            :, self.num_historical_steps : self.num_historical_steps + self.num_future_steps, :2
        ].cpu()
        agent_history_mask = data["agent"]["valid_mask"][:, : self.num_historical_steps].cpu()
        agent_future_mask = data["agent"]["predict_mask"][
            :, self.num_historical_steps : self.num_historical_steps + self.num_future_steps
        ].cpu()
        agent_last_pos = data["agent"]["position"][:, self.num_historical_steps - 1, :2].cpu()

        focal_mask = data["agent"]["category"].cpu() == 3
        focal_idx = int(torch.where(focal_mask)[0][0].item()) if focal_mask.any() else 0

        score_types = []
        for agent_id, category in zip(data["agent"]["id"], data["agent"]["category"].tolist()):
            if agent_id == "AV":
                score_types.append("av")
            elif category == 3:
                score_types.append("focal")
            elif category == 2:
                score_types.append("score")
            elif category == 1:
                score_types.append("unscore")
            else:
                score_types.append("frag")

        agent_types = [_AGENT_TYPES[int(t)] for t in data["agent"]["type"].tolist()]

        return {
            "lane_points": lane_points,
            "agent_history": agent_history,
            "agent_future": agent_future,
            "agent_history_mask": agent_history_mask,
            "agent_future_mask": agent_future_mask,
            "agent_last_pos": agent_last_pos,
            "target_agent_idx": focal_idx,
            "preds": preds[focal_idx],
            "probs": probs[focal_idx],
            "scenario_id": data["scenario_id"],
            "k": self.num_modes,
            "score_types": score_types,
            "log_id": None,
            "agent_types": agent_types,
        }
