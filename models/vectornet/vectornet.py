from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from utils.init_weights import init_weights


class SubgraphEncoder(nn.Module):
    """Encodes individual polylines with a PointNet-style encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.polyline_mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, polyline: torch.Tensor) -> torch.Tensor:
        # polyline: [N, P, F]
        n, p, f = polyline.shape
        x = polyline.reshape(n * p, f)
        x = self.point_mlp(x)
        x = x.reshape(n, p, -1)

        # max-pool across polyline points
        x = torch.max(x, dim=1).values
        return self.polyline_mlp(x)


class EdgeGNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        source_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_feat

        source = node_feat if source_feat is None else source_feat

        return self.propagate(edge_index, x=(source, node_feat), x_dst=node_feat)

    def message(self, x_j):
        return super().message(x_j)

    def update(self, aggr_out, x_dst):
        out = self.update_mlp(torch.cat([x_dst, aggr_out], dim=-1))
        return out


class VectorNetBackbone(nn.Module):
    """Encodes lane and agent polylines and fuses them with graph message passing."""

    def __init__(
        self,
        hidden_dim: int = 128,
        global_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lane_encoder = SubgraphEncoder(2, hidden_dim, hidden_dim, dropout)
        self.agent_encoder = SubgraphEncoder(7, hidden_dim, hidden_dim, dropout)
        self.global_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "lane_lane": EdgeGNNLayer(hidden_dim, dropout),
                        "agent_agent": EdgeGNNLayer(hidden_dim, dropout),
                        "lane_agent": EdgeGNNLayer(hidden_dim, dropout),
                    }
                )
                for _ in range(global_layers)
            ]
        )

        self.apply(init_weights)

    def forward(
        self,
        lane_points: torch.Tensor,
        agent_history: torch.Tensor,
        edge_lane_lane: torch.Tensor,
        edge_agent_agent: torch.Tensor,
        edge_lane_agent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # normalize polylines around their last point for translation invariance
        lane_centroid = lane_points[:, -1:, :2]
        lane_points = lane_points.clone()
        lane_points[:, :, :2] = lane_points[:, :, :2] - lane_centroid

        agent_ref = agent_history[:, -1:, :2]
        agent_points = agent_history.clone()
        agent_points[:, :, :2] = agent_points[:, :, :2] - agent_ref

        lane_feat = self.lane_encoder(lane_points)
        agent_feat = self.agent_encoder(agent_points)

        for layer in self.global_layers:
            lane_feat = lane_feat + layer["lane_lane"](lane_feat, edge_lane_lane)
            agent_feat = agent_feat + layer["agent_agent"](agent_feat, edge_agent_agent)
            # lanes send messages into agents
            agent_feat = agent_feat + layer["lane_agent"](
                agent_feat, edge_lane_agent, source_feat=lane_feat
            )

        return lane_feat, agent_feat


class TrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        future_steps: int,
        dropout: float = 0.1,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.coord_dim = coord_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_steps * coord_dim),
        )

    def forward(self, agent_feat: torch.Tensor) -> torch.Tensor:
        out = self.mlp(agent_feat)
        return out.view(agent_feat.shape[0], self.future_steps, self.coord_dim)

    def loss(self, pred, logits, target):
        loss = F.smooth_l1_loss(pred, target)
        return dict(loss=loss)


class MultiModalTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        future_steps: int,
        dropout: float = 0.1,
        coord_dim: int = 2,
        k: int = 6,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.coord_dim = coord_dim
        self.k = k
        self.traj_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k * future_steps * coord_dim),
        )

        self.prob_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, agent_feat: torch.Tensor) -> torch.Tensor:
        # agent_feat: [B*N]
        out_traj = self.traj_mlp(agent_feat)
        out_traj = out_traj.view(
            agent_feat.shape[0], self.k, self.future_steps, self.coord_dim
        )

        logits = self.prob_mlp(agent_feat)

        return out_traj, logits

    def loss(self, pred, logits, target, lambda_prob=1.0):
        # pred: [num_agents, k, T, 2]
        # logits: [num_agents, k]
        # target: [num_agents, T, 2]
        target_exp = target[:, None, :, :].expand_as(pred)

        per_point = F.smooth_l1_loss(pred, target_exp, reduction="none")  # [n, k, t, 2]
        per_k = per_point.mean(dim=(-1, -2))  # [n, k]

        best_k_indices = per_k.argmin(dim=1)  # [n]

        loss_reg = per_k.gather(1, best_k_indices[:, None]).mean()

        loss_prob = F.cross_entropy(logits, best_k_indices)

        loss = loss_reg + lambda_prob * loss_prob

        return dict(loss_reg=loss_reg, loss_prob=loss_prob, loss=loss)


class VectorNetTrajPred(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        global_layers: int = 1,
        dropout: float = 0.1,
        future_steps: int = 30,
        k: int = 1,
    ):
        super().__init__()
        self.k = k
        self.backbone = VectorNetBackbone(
            hidden_dim=hidden_dim, global_layers=global_layers, dropout=dropout
        )
        self.decoder = (
            TrajectoryDecoder(
                hidden_dim=hidden_dim, future_steps=future_steps, dropout=dropout
            )
            if k == 1
            else MultiModalTrajectoryDecoder(
                hidden_dim=hidden_dim, future_steps=future_steps, dropout=dropout, k=k
            )
        )

    def forward(self, batch: dict) -> torch.Tensor:
        lane_feat, agent_feat = self.backbone(
            lane_points=batch["lane_points"],
            agent_history=batch["agent_history"],
            edge_lane_lane=batch["edge_index_lane_to_lane"],
            edge_agent_agent=batch["edge_index_agent_to_agent"],
            edge_lane_agent=batch["edge_index_lane_to_agent"],
        )

        target_feat = agent_feat[batch["target_agent_global_idx"]]  # [B, C]
        if self.k == 1:
            # single agent single modal
            pred = self.decoder(target_feat)
            pred = pred + batch["target_last_pos"].unsqueeze(1)
            return pred, None
        else:
            # single agent multi modal
            pred, logits = self.decoder(target_feat)
            pred = pred + batch["target_last_pos"][:, None, None, :]
            return pred, logits

    def loss(self, pred, logits, batch) -> dict:
        return self.decoder.loss(pred, logits, batch["target_gt"])

    def select_best_modal(self, pred, logits) -> torch.Tensor:
        # preds: [n, k , t, 2]
        # logits: [n, k]

        probs = F.softmax(logits, dim=1)  # [n, k]
        best_k = probs.argmax(dim=1)  # [n]
        n = pred.shape[0]
        best_pred = pred[torch.arange(n, device=pred.device), best_k]

        return best_pred
