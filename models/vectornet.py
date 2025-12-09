from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphEncoder(nn.Module):
    """Encodes individual polylines with a PointNet-style encoder."""

    def __init__(
        self, hidden_dim: int = 128, output_dim: int = 128, dropout: float = 0.1
    ):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.polyline_mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
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


class EdgeGNNLayer(nn.Module):
    """Message passing layer using averaged neighbor aggregation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.update_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

        src, dst = edge_index
        source = node_feat if source_feat is None else source_feat

        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, dst, source[src])

        deg = torch.zeros(
            node_feat.shape[0], device=node_feat.device, dtype=node_feat.dtype
        )
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=node_feat.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)

        return self.update_mlp(torch.cat([node_feat, agg], dim=-1))


class VectorNetBackbone(nn.Module):
    """Encodes lane and agent polylines and fuses them with graph message passing."""

    def __init__(
        self,
        hidden_dim: int = 128,
        global_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lane_encoder = SubgraphEncoder(hidden_dim, hidden_dim, dropout)
        self.agent_encoder = SubgraphEncoder(hidden_dim, hidden_dim, dropout)
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_steps * coord_dim),
        )

    def forward(self, agent_feat: torch.Tensor) -> torch.Tensor:
        out = self.mlp(agent_feat)
        return out.view(agent_feat.shape[0], self.future_steps, self.coord_dim)


class VectorNetTrajPred(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        global_layers: int = 1,
        dropout: float = 0.1,
        future_steps: int = 30,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = VectorNetBackbone(
            hidden_dim=hidden_dim, global_layers=global_layers, dropout=dropout
        )
        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim, future_steps=future_steps, dropout=dropout
        )
        self.lr = lr

    def forward(self, batch: dict) -> torch.Tensor:
        lane_feat, agent_feat = self.backbone(
            lane_points=batch["lane_points"],
            agent_history=batch["agent_history"],
            edge_lane_lane=batch["edge_index_lane_to_lane"],
            edge_agent_agent=batch["edge_index_agent_to_agent"],
            edge_lane_agent=batch["edge_index_lane_to_agent"],
        )

        target_feat = agent_feat[batch["target_agent_global_idx"]]
        pred = self.decoder(target_feat)
        return pred + batch["target_last_pos"].unsqueeze(1)

    def training_step(self, batch: dict, batch_idx: int):
        pred = self(batch)
        loss = F.smooth_l1_loss(pred, batch["target_gt"])
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["target_gt"].shape[0],
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        pred = self(batch)
        loss = F.smooth_l1_loss(pred, batch["target_gt"])
        self.log(
            "val/loss", loss, prog_bar=True, batch_size=batch["target_gt"].shape[0]
        )

    def test_step(self, batch: dict, batch_idx: int):
        pred = self(batch)
        loss = F.smooth_l1_loss(pred, batch["target_gt"])
        self.log("test/loss", loss, batch_size=batch["target_gt"].shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
