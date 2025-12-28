from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional

import torch
import numpy as np

import matplotlib

matplotlib.use("Agg")  # safe headless backend for dataloader forks


from utils.numpy import to_numpy

_COLOR_MAP = {
    # 格式：[历史轨迹, 未来/真值轨迹, 当前位置/预测点]
    
    # Focal Agent (红色系：强调高对比度)
    # 历史用淡红，未来用鲜红，当前点用深红/红点
    "focal": ["#ff9999", "#ff0000", "#8b0000"],
    
    # AV (蓝色系：专业且冷色调)
    # 历史浅蓝，未来中蓝，当前位置深蓝
    "av": ["#a1c9f4", "#1f77b4", "#084594"],
    
    # Score/Prediction (橙色系：警示色，用于区分真值和预测)
    # 预测轨迹通常建议用虚线或明亮的橙色
    "score": ["#ffbb78", "#ff7f0e", "#a65628"],
    
    # Other Agents (灰色系：背景化，减少视觉干扰)
    "other": ["#d3d3d3", "#7f7f7f", "#555555"],
    
    # 地图元素 (极浅灰：仅作为参考坐标)
    "lane": "#e0e0e0",
}

_POINT_SIZE = 5
_LINE_WIDTH = 1.0
_FONT_SIZE = 6

_VIEW_RADIUS = 60.0

_SCORE_TYPES = [
    "focal",
    "av",
    "score",
    # "unscore",
    # "frag",
]


def plot_scenario(
    lane_points: torch.Tensor | np.ndarray,
    agent_history: torch.Tensor | np.ndarray,
    agent_future: torch.Tensor | np.ndarray,
    agent_history_mask: torch.Tensor | np.ndarray,
    agent_future_mask: torch.Tensor | np.ndarray,
    agent_last_pos: torch.Tensor | np.ndarray,
    target_agent_idx: int,
    preds: torch.Tensor | np.ndarray | None = None,
    probs: torch.Tensor | None = None,
    scenario_id: str | None = None,
    k: int = 1,
    only_print_focal_pred: bool = False,
    score_types: list[str] | None = None,
):
    """Plot lanes, agent history, target future ground truth, and prediction."""

    lane_points_np = to_numpy(lane_points)  # (num_lanes, num_points, 2)
    agent_history_np = to_numpy(agent_history)  # (num_agents, hist_len, 7)
    agnet_history_mask_np = to_numpy(agent_history_mask)  # (num_agents, hist_len)
    agent_future_np = to_numpy(agent_future)  # (num_agents, fut_len, 2)
    agent_future_mask_np = to_numpy(agent_future_mask)  # (num_agents, fut_len)
    agent_last_pos_np = to_numpy(agent_last_pos)  # (num_agents, 2)
    preds_np = to_numpy(preds) if preds is not None else None
    probs_np = to_numpy(probs) if probs is not None else None

    valid_agents = agnet_history_mask_np.any(-1)
    valid_indices = np.where(valid_agents)[0]
    num_agents = valid_agents.sum()
    agent_history_np = agent_history_np[valid_agents]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # plot lanes
    ax = _plot_lanes(ax, lane_points_np)

    # plot agents
    for idx in valid_indices:
        agent_type = score_types[idx]
        agent_label = None
        current_only = True

        if agent_type == "focal":
            agent_type = "focal"
            agent_label = "focal"
            current_only = False
        elif agent_type == "av":
            agent_type = "av"
            agent_label = "av"
            current_only = False
        elif agent_type == "score":
            agent_type = "score"
            agent_label = "scored"
            current_only = False
        else:
            agent_type = "other"
            agent_label = None
            current_only = True

        ax = _plot_agent(
            ax,
            agent_history_np[idx],
            agent_future_np[idx],
            agnet_history_mask_np[idx],
            agent_future_mask_np[idx],
            agent_last_pos_np[idx],
            color_map=_COLOR_MAP[agent_type],
            label=agent_label,
            current_only=current_only,
        )

    # plot predictions
    if preds_np.ndim == 4:
        # [n, k, t, 2]
        for idx in range(num_agents):
            agent_type = score_types[idx]
            if agent_type not in _SCORE_TYPES:
                continue
            ax = _plot_predictions(
                ax,
                preds_np[idx],
                probs_np[idx] if probs_np is not None else None,
                max_k=k,
                color_map=_COLOR_MAP[agent_type],
                plot_text=(agent_type in ["focal", "av"]),
            )
    elif preds_np.ndim == 3:
        # [k, t, 2]
        ax = _plot_predictions(
            ax,
            preds_np,
            probs_np if probs_np is not None else None,
            max_k=k,
            color_map=_COLOR_MAP["focal"],
        )

    # if target_last_pos_np is not None:
    cx, cy = agent_last_pos_np[target_agent_idx]

    ax.set_xlim(cx - _VIEW_RADIUS, cx + _VIEW_RADIUS)
    ax.set_ylim(cy - _VIEW_RADIUS, cy + _VIEW_RADIUS)

    ax.set_aspect("equal", adjustable="box")

    if scenario_id is not None:
        title = f"Log: {scenario_id}"
        ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def _plot_lanes(
    ax: plt.Axes,
    lane_points_np: np.ndarray,
):
    # Map polylines
    for lane in lane_points_np:
        ax.plot(
            lane[:, 0],
            lane[:, 1],
            color=_COLOR_MAP["lane"],
            linewidth=_LINE_WIDTH,
            zorder=0,
        )

    return ax


def _plot_agent(
    ax: plt.Axes,
    agent_history_np: np.ndarray,
    agent_future_np: np.ndarray,
    agent_history_mask_np: np.ndarray,
    agent_future_mask_np: np.ndarray,
    agent_last_pos_np: np.ndarray,
    color_map: Optional[dict] = _COLOR_MAP["other"],
    label: Optional[str] = None,
    current_only: bool = True,
):

    if not current_only:
        traj_history = agent_history_np[agent_history_mask_np]
        traj_future = agent_future_np[agent_future_mask_np]

        ax.plot(
            traj_history[:, 0],
            traj_history[:, 1],
            color=color_map[0],
            linewidth=_LINE_WIDTH,
            label=f"{label} history" if label is not None else None,
            zorder=1,
        )
        ax.plot(
            traj_future[:, 0],
            traj_future[:, 1],
            color=color_map[1],
            linewidth=_LINE_WIDTH,
            # linestyle="--",
            label=f"{label} future" if label is not None else None,
            zorder=2,
        )
    ax.scatter(
        agent_last_pos_np[0],
        agent_last_pos_np[1],
        color=color_map[2],
        s=_POINT_SIZE,
        zorder=5,
    )
    return ax


def _plot_predictions(
    ax: plt.Axes,
    preds_np: np.ndarray,
    probs_np: np.ndarray,
    max_k: int | None = None,
    color_map: Optional[dict] = _COLOR_MAP["other"],
    plot_text: bool = False,
):
    # preds_np.shape [k, t, 2]
    # probs_np.shape [k]
    k = preds_np.shape[0]
    color = color_map[2]

    if max_k is not None and max_k < k:
        sorted_indices = np.argsort(-probs_np)  # descending
        selected_indices = sorted_indices[:max_k]
        preds_np = preds_np[selected_indices]
        probs_np = probs_np[selected_indices]

    for pred_coords, pred_prob in zip(preds_np, probs_np):
        # ax.scatter(
        #     pred_coords[:, 0],
        #     pred_coords[:, 1],
        #     c=color,
        #     s=_POINT_SIZE,
        #     alpha=0.5,
        # )
        ax.plot(
            pred_coords[:, 0],
            pred_coords[:, 1],
            color=color,
            linewidth=_LINE_WIDTH,
            linestyle="--",
            alpha=0.3,
            zorder=3,
        )
        ax.scatter(
            pred_coords[-1, 0],
            pred_coords[-1, 1],
            color=color,
            marker="o",
            s=_POINT_SIZE,
            alpha=0.3,
            zorder=5,
        )
        
        

        # text
        if plot_text:
            ax.text(
                pred_coords[-1, 0],
                pred_coords[-1, 1],
                f"{pred_prob:.2f}",
                color=color,
                fontsize=_FONT_SIZE,
            )

    return ax
