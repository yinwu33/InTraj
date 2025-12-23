from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional

import torch
import numpy as np

import matplotlib

matplotlib.use("Agg")  # safe headless backend for dataloader forks


from utils.numpy import to_numpy

_COLOR_MAP = {
    # agent: history, future, last_position
    # blue, green, red series
    "focal": ["#1f77b4", "#2ca02c", "#d62728"],
    "ego": ["#17becf", "#98df8a", "#d62728"],
    "other": ["#7f7f7f", "#9ed9a0", "#f7b6d2"],
    "lane": "#c7c7c7",
}


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
    view_radius: float = 80.0,
    k: int = 1,
):
    """Plot lanes, agent history, target future ground truth, and prediction."""

    lane_points_np = to_numpy(lane_points)  # (num_lanes, num_points, 2)
    agent_history_np = to_numpy(agent_history)  # (num_agents, hist_len, 7)
    agent_future_np = to_numpy(agent_future)  # (num_agents, fut_len, 2)
    agnet_history_mask_np = to_numpy(agent_history_mask)  # (num_agents, hist_len)
    agent_future_mask_np = to_numpy(agent_future_mask)  # (num_agents, fut_len)
    agent_last_pos_np = to_numpy(agent_last_pos)  # (num_agents, 2)
    preds_np = to_numpy(preds) if preds is not None else None
    probs_np = to_numpy(probs) if probs is not None else None

    fig, ax = plt.subplots(figsize=(6, 6))

    ax = _plot_lanes(ax, lane_points_np)

    for idx in range(agent_history_np.shape[0]):
        agent_type = "other"
        agent_label = None
        if idx == target_agent_idx:
            agent_type = "focal"
            agent_label = "focal"
        elif idx == 1:
            agent_type = "ego"
            agent_label = "ego"
        else:
            agent_type = "other"
            agent_label = None
        ax = _plot_agent(
            ax,
            agent_history_np[idx],
            agent_future_np[idx],
            agnet_history_mask_np[idx],
            agent_future_mask_np[idx],
            agent_last_pos_np[idx],
            color_map=_COLOR_MAP[agent_type],
            label=agent_label,
        )

    # plot predictions
    if preds_np.ndim == 4:
        # [n, k, t, 2]
        for idx in range(agent_history_np.shape[0]):
            agent_type = "other"
            if idx == target_agent_idx:
                agent_type = "focal"
            else:
                agent_type = "other"
            ax = _plot_predictions(
                ax,
                preds_np[idx],
                probs_np[idx] if probs_np is not None else None,
                max_k=k,
                color_map=_COLOR_MAP[agent_type],
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

    ax.set_xlim(cx - view_radius, cx + view_radius)
    ax.set_ylim(cy - view_radius, cy + view_radius)

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
            lane[:, 0], lane[:, 1], color=_COLOR_MAP["lane"], linewidth=1.0, zorder=0
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
):

    traj_history = agent_history_np[agent_history_mask_np]
    traj_future = agent_future_np[agent_future_mask_np]

    ax.plot(
        traj_history[:, 0],
        traj_history[:, 1],
        color=color_map[0],
        linewidth=1.0,
        label=f"{label} history" if label is not None else None,
    )
    ax.plot(
        traj_future[:, 0],
        traj_future[:, 1],
        color=color_map[1],
        linewidth=1.0,
        label=f"{label} future" if label is not None else None,
    )
    ax.scatter(
        agent_last_pos_np[0],
        agent_last_pos_np[1],
        color=color_map[2],
        s=10,
    )
    return ax


def _plot_predictions(
    ax: plt.Axes,
    preds_np: np.ndarray,
    probs_np: np.ndarray,
    max_k: int | None = None,
    color_map: Optional[dict] = _COLOR_MAP["other"],
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
        ax.plot(
            pred_coords[:, 0],
            pred_coords[:, 1],
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=pred_prob,
        )

    return ax
