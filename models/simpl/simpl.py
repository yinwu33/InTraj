import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
print(f"Root path added: {root_path}")
sys.path.append(str(root_path))

from utils.init_weights import init_weights
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from torch import Tensor, nn
import torch
from math import gcd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from typing import Any, Dict, List, Tuple, Union, Optional
from .loss import LossFunc


class Conv1d(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=3,
        stride=1,
        norm="GN",
        num_groups=32,
        act=True,
    ):
        super(Conv1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(num_groups, output_dim), output_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(output_dim)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=3,
        stride=1,
        norm="GN",
        num_groups=32,
        act=True,
    ):
        super(Res1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            output_dim,
            output_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(num_groups, output_dim), output_dim)
            self.bn2 = nn.GroupNorm(gcd(num_groups, output_dim), output_dim)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(output_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or output_dim != input_dim:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        input_dim, output_dim, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.GroupNorm(gcd(num_groups, output_dim), output_dim),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        input_dim, output_dim, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm1d(output_dim),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Extract actor feature from timeseries of actor states
    (num_actors, C_in, L_in) -> (num_actors, hidden_dim)
    """

    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim

        self.norm = "GN"
        self.num_groups = 1

        # output channel for each stage: [32, 64, 128]
        channels_list = [2 ** (5 + s) for s in range(cfg.num_fpn_scale)]

        # * build encoder stages: (B, C_in, L) -> (B, C_out, L_out)
        # e.g. 4 stages, (B, C_in, L) -> (B, 32, L) -> (B, 64, L/2) -> (B, 128, L/4) -> (B, 256, L/8)
        self.encoder_stages = nn.ModuleList()
        current_in = cfg.input_dim
        for i, out_channel in enumerate(channels_list):
            stride = 1 if i == 0 else 2
            stage = self._make_stage(current_in, out_channel, stride, 2)
            self.encoder_stages.append(stage)
            current_in = out_channel

        # * build lateral
        self.lateral_convs = nn.ModuleList()
        for out_channel in channels_list:
            lateral_conv = Conv1d(
                out_channel,
                cfg.hidden_dim,
                norm=self.norm,
                num_groups=self.num_groups,
                act=False,
            )
            self.lateral_convs.append(lateral_conv)

        # * build output head
        self.output_head = Res1d(
            cfg.hidden_dim, cfg.hidden_dim, norm=self.norm, num_groups=self.num_groups
        )

    def _make_stage(self, in_channels, out_channels, stride, num_blocks):
        """
        build a resnet stage with `num_blocks` blocks
        # input: Tensor of shape (B, C_in, L)
        # output: Tensor of shape (B, C_out, L) for stride=1, (B, C_out, L/2) for stride=2
        """
        layers = []
        # first: in_channels -> out_channels
        layers.append(
            Res1d(
                in_channels,
                out_channels,
                stride=stride,
                norm=self.norm,
                num_groups=self.num_groups,
            )
        )

        # rest: out_channels -> out_channels
        for _ in range(1, num_blocks):
            layers.append(
                Res1d(
                    out_channels,
                    out_channels,
                    stride=1,
                    norm=self.norm,
                    num_groups=self.num_groups,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, agent_feats: Tensor) -> Tensor:
        # actor: (num_actors, C_in, L_in): (num_actors, channel, timesteps)
        # output: (num_actors, hidden_dim)
        # agent_feats: (B, Nmax, D, T)
        features = []

        B, N, D, T = agent_feats.shape
        x = agent_feats.reshape(B * N, D, T)

        for stage in self.encoder_stages:
            x = stage(x)
            features.append(x)

        # now x is (B, 256, L/8) if num_fpn_scale=4
        # features is [(B, 32, L), (B, 64, L/2), (B, 128, L/4), (B, 256, L/8)]

        # (B, hidden_dim, L/8)
        prev_feature = self.lateral_convs[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            lateral_feature = self.lateral_convs[i](
                features[i]
            )  # (B, hidden_dim, L/(2^i))
            upsampled_feature = F.interpolate(
                prev_feature,
                size=lateral_feature.shape[-1],
                mode="linear",
                align_corners=False,
            )  #
            prev_feature = (
                lateral_feature + upsampled_feature
            )  # (B, hidden_dim, L/(2^i))

        out = self.output_head(prev_feature)  # (B, hidden_dim, L)

        out = out.reshape(B, N, *out.shape[1:])

        return out[..., -1]  # (B, N, H)


class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),  # ! this is activated in original code
            # nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _aggregate(self, feat):
        # feat: (n, 10, h) -> (n, h, 10) -> (n, h, 1) -> (n, 1, h)
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x):
        # x: (num_lanes, 10, hidden_size)
        feat = self.fc1(x)  # (num_lanes, 10, hidden_size)

        x_aggre = self._aggregate(feat)  # (num_lanes, 1, hidden_size)
        combined = torch.cat([feat, x_aggre.expand_as(feat)], dim=-1)  # (n, 10, h*2)

        out = self.norm(x + self.fc2(combined))

        if self.aggre_out:
            return self._aggregate(out).squeeze()  # (n, h)
        return out  # (n, 10, h)


class LaneNet(nn.Module):
    """Extract lane feature from lane centerline polylines
    (num_lanes, 10, C_in) -> (num_lanes, hidden_dim)"""

    def __init__(self, cfg):
        super(LaneNet, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.aggre1 = PointAggregateBlock(
            hidden_size=cfg.hidden_dim, aggre_out=False, dropout=cfg.dropout
        )
        self.aggre2 = PointAggregateBlock(
            hidden_size=cfg.hidden_dim, aggre_out=True, dropout=cfg.dropout
        )

    def forward(self, x):
        # input: (num_lanes, 10, input_dim)
        B, N, D, T = x.shape
        x = x.reshape(B * N, D, T)
        feats = self.proj(x)  # (num_lanes, 10, hidden_size)
        feats = self.aggre1(feats)  # (num_lanes, 10, hidden_size)
        feats = self.aggre2(feats)  # (num_lanes, hidden_size)
        feats = feats.reshape(B, N, -1)
        return feats


class SftLayer(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        fnn_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        update_edge: bool = True,
    ) -> None:
        super(SftLayer, self).__init__()
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(inplace=True),
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(node_dim, edge_dim),
                nn.LayerNorm(edge_dim),
                nn.ReLU(inplace=True),
            )
            self.norm_edge = nn.LayerNorm(edge_dim)

        self.multihead_attn = MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Feedforward model
        self.linear1 = nn.Linear(node_dim, fnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fnn_dim, node_dim)

        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self, node: Tensor, edge: Tensor, node_mask: Tensor, edge_mask: Tensor
    ) -> Tensor:
        # node: [B, Nmax, D]
        # edge: [B, Nmax, Nmax, D]
        # node_mask: [B, Nmax]
        # edge_mask: [B, Nmax, Nmax]

        B, N, D = node.shape

        # x: node: [B, Nmax, D]
        # edge: node: [B, Nmax, Nmax, D]
        # memory: node: [B, Nmax, Nmax, D]
        x, edge, memory = self._build_memory(node, edge)

        # reshape for multihead attention
        x_mha = x.reshape(1, B * N, D)  # [1, BN, D]
        memory_mha = memory.permute(2, 0, 1, 3).reshape(N, B * N, D)  # [N, BN, D]

        mask_mha = node_mask.unsqueeze(1).expand(B, N, N).reshape(B * N, N)

        x_prime, _ = self._mha_block(
            x_mha, memory_mha, attn_mask=None, key_padding_mask=~mask_mha
        )

        # 4. Reshape back & Residual
        # (1, B*N, D) -> (B, N, D)
        x_prime = x_prime.reshape(B, N, D)

        # Residual + Norm
        x = self.norm2(node + x_prime)
        x = self.norm3(x + self._ff_block(x))

        return x, edge, None

    def _build_memory(
        self,
        node: Tensor,
        edge: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        input:
            node:   (B, N, d_model)
            edge:   (B, N, N, d_edge)
            node_mask: (B, N)
        output:
            :param  (1, N, d_model)
            :param  (N, N, d_edge)
            :param  (N, N, d_model)
        """
        B, N, D = node.shape

        # 1. build memory
        src_x = node[:, :, None, :].expand([B, N, N, D])
        tar_x = node[:, None, :, :].expand([B, N, N, D])
        memory = self.proj_memory(
            torch.cat([edge, src_x, tar_x], dim=-1)
        )  # [B, N, N, D]
        # memory = memory.masked_fill(~node_mask[..., None], 0.0)

        # 2. (optional) update edge (with residual)
        if self.update_edge:
            # TODO: add after relu is correct?
            # TODO: masked_fill?
            edge = self.norm_edge(edge + self.proj_edge(memory))

        return node, edge, memory

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """
        input:
            x:                  [1, N, d_model]
            mem:                [N, N, d_model]
            attn_mask:          [N, N]
            key_padding_mask:   [N, N]
        output:
            :param      [1, N, d_model]
            :param      [N, N]
        """
        x, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        update_edge: bool = True,
    ):
        super().__init__()

        fusion = []
        for i in range(num_layers):
            need_update_edge = False if i == num_layers - 1 else update_edge
            fusion.append(
                SftLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    fnn_dim=node_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                    update_edge=need_update_edge,
                )
            )
        self.fusion = nn.ModuleList(fusion)

    def forward(
        self, feats: Tensor, rpe: Tensor, feats_mask: Tensor, rpe_mask
    ) -> Tensor:
        """
        feats: (B, N, D)
        masks: (B, N)
        rpe: (B, N, N, D)
        """
        # attn_multilayer = []
        for mod in self.fusion:
            feats, masks, _ = mod(feats, rpe, feats_mask, rpe_mask)  # TODO:  mask?
            # attn_multilayer.append(attn)
        return feats


class FusionNet(nn.Module):
    def __init__(self, cfg):
        super(FusionNet, self).__init__()
        self.proj_actor = nn.Sequential(
            nn.Linear(cfg.actor_emb_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(cfg.lane_emb_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(cfg.rpe_input_dim, cfg.rpe_emb_dim),
            nn.LayerNorm(cfg.rpe_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.fuse_scene = SymmetricFusionTransformer(
            node_dim=cfg.hidden_dim,
            edge_dim=cfg.rpe_emb_dim,
            num_heads=cfg.num_scene_heads,
            num_layers=cfg.num_scene_layers,
            dropout=cfg.dropout,
            update_edge=cfg.update_edge,
        )

    def forward(
        self,
        agent_feats: Tensor,
        agent_masks: Tensor,
        lane_feats: Tensor,
        lane_masks: Tensor,
        rpe_feats: Tensor,
        rpe_masks: Tensor,
    ):
        """
        agent_feats (Tensor): [B, N_a, D]
        agent_masks (Tensor): [B, N_a], bool
        lane_feats (Tensor): [B, N_l, D]
        lane_masks (Tensor): [B, N_l] bool
        rpe_feats (Tensor): list of (5, n, n) n is the total number of agent and lanes in each sample
        rpe_masks (Tensor): [B, N, N] bool
        """
        B, N_a, D = agent_feats.shape
        _, N_l, _ = lane_feats.shape
        N = N_a + N_l

        # projection
        agent_feats = self.proj_actor(agent_feats.reshape(B * N_a, D))
        agent_feats = agent_feats.reshape(B, N_a, -1)
        agent_feats = agent_feats.masked_fill(~agent_masks[..., None], 0.0)

        lane_feats = self.proj_lane(lane_feats.reshape(B * N_l, D))
        lane_feats = lane_feats.reshape(B, N_l, -1)
        lane_feats = lane_feats.masked_fill(~lane_masks[..., None], 0.0)

        feats = torch.concat([agent_feats, lane_feats], dim=1)  # [B, Nmax, D]
        feat_masks = torch.concat([agent_masks, lane_masks], dim=1)  # [B, Nmax]

        rpe_feats = self.proj_rpe_scene(rpe_feats.permute(0, 2, 3, 1))  # (b, n, n, d)
        rpe_feats = rpe_feats.masked_fill(~rpe_masks[..., None], 0.0)

        out = self.fuse_scene(feats, rpe_feats, feat_masks, rpe_masks)

        return out[:, :N_a, :], out[:, N_a:, :]


class MLPDecoder(nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super(MLPDecoder, self).__init__()
        # self.device = device
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.future_steps = cfg.global_pred_lane
        self.k = cfg.k
        # parametric output: bezier/monomial/none
        self.param_out = cfg.param_out
        self.N_ORDER = cfg.param_order

        dim_mm = self.hidden_dim * self.k
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

        if self.param_out == "bezier":
            self.register_buffer(
                "mat_T",
                self._get_T_matrix_bezier(
                    n_order=self.N_ORDER, n_step=self.future_steps
                ),
            )

            self.register_buffer(
                "mat_Tp",
                self._get_Tp_matrix_bezier(
                    n_order=self.N_ORDER, n_step=self.future_steps
                ),
            )

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, (self.N_ORDER + 1) * 2),
            )
        elif self.param_out == "monomial":
            self.mat_T = self._get_T_matrix_monomial(
                n_order=self.N_ORDER, n_step=self.future_steps
            )  # .to(self.device)
            self.mat_Tp = self._get_Tp_matrix_monomial(
                n_order=self.N_ORDER, n_step=self.future_steps
            )  # .to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, (self.N_ORDER + 1) * 2),
            )
        elif self.param_out == "none":
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.future_steps * 2),
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts) ** (n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        # ! NOTICE: we multiply n_order inside of the Tp matrix
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (
                n_order
                * math.comb(n_order - 1, i)
                * (1.0 - ts) ** (n_order - 1 - i)
                * ts**i
            )
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts**i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(
        self, agent_feats: torch.Tensor, agent_masks: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = agent_feats.shape
        agent_feats = agent_feats.reshape(-1, H)

        agent_feats = (
            self.multihead_proj(agent_feats).view(-1, self.k, H).permute(1, 0, 2)
        )  # [k, BN, H]

        cls = self.cls(agent_feats).view(self.k, -1).permute(1, 0)  # [BN, k]
        cls = F.softmax(cls * 1.0, dim=1)  # [BN, k]
        cls = cls.reshape(B, N, self.k)

        if self.param_out == "bezier":
            param = self.reg(agent_feats).view(self.k, -1, self.N_ORDER + 1, 2)
            param = param.permute(1, 0, 2, 3)  # e.g., [BN, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [BN, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, torch.diff(param, dim=2)) / (
                self.future_steps * 0.1
            )
        elif self.param_out == "monomial":
            # e.g., [6, BN, N_ORDER + 1, 2]
            param = self.reg(agent_feats).view(self.k, -1, self.N_ORDER + 1, 2)
            param = param.permute(1, 0, 2, 3)  # e.g., [BN, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [BN, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (
                self.future_steps * 0.1
            )
        elif self.param_out == "none":
            reg = self.reg(agent_feats).view(
                self.k, -1, self.future_steps, 2
            )  # e.g., [6, BN, 30, 2]
            reg = reg.permute(1, 0, 2, 3)  # e.g., [BN, 6, 30, 2]
            # vel is calculated from pos
            vel = torch.gradient(reg, dim=-2)[0] / 0.1

        reg = reg.reshape(B, N, *reg.shape[1:])
        vel = vel.reshape(B, N, *vel.shape[1:])
        param = param.reshape(B, N, *param.shape[1:])

        # print('reg: ', reg.shape, 'cls: ', cls.shape)
        # de-batchify
        res_cls, res_reg, res_aux = [], [], []
        # for i in range(len(actor_idcs)):
        for i in range(B):
            res_cls.append(cls[i][agent_masks[i]])
            res_reg.append(reg[i][agent_masks[i]])

            if self.param_out == "none":
                res_aux.append(
                    (vel[i][agent_masks[i]], None)
                )  # ! None is a placeholder
            else:
                # List[Tuple[Tensor,...]]
                res_aux.append((vel[i][agent_masks[i]], param[i][agent_masks[i]]))

        return res_cls, res_reg, res_aux


class Simpl(nn.Module):
    # Initialization
    def __init__(
        self,
        actor_net_cfg,
        lane_net_cfg,
        fusion_net_cfg,
        mlp_decoder_cfg,
        loss_cfg,
    ):
        super(Simpl, self).__init__()

        # self.rpe_embedding = RPEEmbedding()  # TODO: remove

        self.actor_net = ActorNet(actor_net_cfg)
        self.lane_net = LaneNet(lane_net_cfg)
        self.fusion_net = FusionNet(fusion_net_cfg)
        self.pred_net = MLPDecoder(mlp_decoder_cfg)
        self.loss = LossFunc(loss_cfg)

        self.k = mlp_decoder_cfg.k

        self.apply(init_weights)

    def forward(self, data):
        # actors, actor_idcs, lanes, lane_idcs, rpe = data
        agent_inputs = data["agent_history"].permute([0, 1, 3, 2])  # [B, Nmax, D, T]
        agent_history_masks = data["agent_history_mask"]  # [B, Nmax]
        agent_masks = agent_history_masks.any(dim=-1)  # [B, Nmax]
        lane_inputs = data["lane_feats"]  # [B, Nmax, 10, 16]
        lane_masks = data["lane_masks"]  # [B, Nmax]
        rpe_list = data["rpe"]  # list of (5, n, n)

        B = agent_inputs.shape[0]
        N_a = agent_inputs.shape[1]
        N_l = lane_inputs.shape[1]
        N = N_a + N_l

        rpe_inputs = torch.zeros(
            [B, 5, N, N], dtype=rpe_list[0].dtype, device=rpe_list[0].device
        )
        rpe_masks = torch.zeros([B, N, N], dtype=torch.bool, device=rpe_list[0].device)
        for i in range(B):
            num_all = rpe_list[i].shape[1]
            rpe_inputs[i, :, :num_all, :num_all] = rpe_list[i]
            rpe_masks[i, :num_all, :num_all] = True

        # * actors/lanes encoding
        agent_feats = self.actor_net(agent_inputs)  # (b, n, d, t) -> (b, n, hidden_dim)
        lane_feats = self.lane_net(lane_inputs)  # (b, n, d, t) -> (b, n, hidden_dim)
        # * fusion
        agent_feats, lane_feats = self.fusion_net(
            agent_feats,
            agent_masks,
            lane_feats,
            lane_masks,
            rpe_inputs,
            rpe_masks,
        )
        # * decoding
        out = self.pred_net(agent_feats, agent_masks)

        return out

    def post_process(self, out):
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out["out_raw"] = out
        post_out["traj_pred"] = reg  # batch x n_mod x pred_len x 2
        post_out["prob_pred"] = cls  # batch x n_mod

        return post_out
