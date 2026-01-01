from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional
from .viz_av2 import AV2MapVisualizer

import torch
import numpy as np

import matplotlib

matplotlib.use("Agg")  # safe headless backend for dataloader forks


from utils.numpy import to_numpy

_COLOR_MAP = {
    "focal": ["#ff9999", "#ff0000", "#8b0000"],
    "av": ["#a1c9f4", "#1f77b4", "#084594"],
    "score": ["#ffbb78", "#ff7f0e", "#a65628"],
    "unscore": ["#d3d3d3", "#7f7f7f", "#555555"],
    "frag": ["#d3d3d3", "#7f7f7f", "#555555"],
    "lane": "#e0e0e0",
}

_ID_TO_SCORE_TYPE = {
    4: "av",
    3: "focal",
    2: "score",
    1: "unscore",
    0: "frag",
}


_POINT_SIZE = 5
_LINE_WIDTH = 1.0
_FONT_SIZE = 6

_VIEW_RADIUS = 60.0

_SCORE_TYPES = ["focal", "av", "score"]

_AGENT_BBOX = {
    "vehicle": (5.0, 2.0),
    "motorcyclist": (2.0, 0.7),
    "pedestrian": (0.3, 0.5),
    "cyclist": (2.0, 0.7),
    "bus": (7.0, 2.1),
    # "unknown": None,
    # "default": None,
}


def plot_scenario(
    lane_points: torch.Tensor | np.ndarray,
    agent_hist_pos: torch.Tensor | np.ndarray,
    agent_fut_pos: torch.Tensor | np.ndarray,
    agent_hist_mask: torch.Tensor | np.ndarray,
    agent_fut_mask: torch.Tensor | np.ndarray,
    agent_last_pos: torch.Tensor | np.ndarray,
    # agent_last_ang: torch.Tensor | np.ndarray,
    target_agent_idx: int,
    preds: torch.Tensor | np.ndarray | None = None,
    probs: torch.Tensor | None = None,
    scenario_id: str | None = None,
    k: int = 1,
    score_types: list[str] | None = None,
    log_id: str = None,
    agent_types: list[str] | None = None,
):
    """Plot lanes, agent history, target future ground truth, and prediction."""

    lane_points_np = to_numpy(lane_points)  # (num_lanes, num_points, 2)
    agent_history_np = to_numpy(agent_hist_pos)  # (num_agents, hist_len, 7)
    agnet_history_mask_np = to_numpy(agent_hist_mask)  # (num_agents, hist_len)
    agent_future_np = to_numpy(agent_fut_pos)  # (num_agents, fut_len, 2)
    agent_future_mask_np = to_numpy(agent_fut_mask)  # (num_agents, fut_len)
    agent_last_pos_np = to_numpy(agent_last_pos)  # (num_agents, 2)
    preds_np = to_numpy(preds) if preds is not None else None
    probs_np = to_numpy(probs) if probs is not None else None

    valid_agents = agnet_history_mask_np.any(-1)
    valid_indices = np.where(valid_agents)[0]
    num_agents = valid_agents.sum()
    agent_history_np = agent_history_np[valid_agents]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # plot lanes
    ax = _plot_lanes(ax, lane_points_np, log_id)

    # plot agents
    for idx in valid_indices:
        agent_score_type = _ID_TO_SCORE_TYPE[int(score_types[idx])]
        agent_label = None
        current_only = True
        agent_type = agent_types[idx]

        if agent_score_type == "focal":
            agent_label = "focal"
            current_only = False
        elif agent_score_type == "av":
            agent_label = "av"
            current_only = False
        elif agent_score_type == "score":
            agent_label = "score"
            current_only = False
        else:
            agent_label = None
            current_only = True

        ax = _plot_agent(
            ax,
            agent_history_np[idx],
            agent_future_np[idx],
            agnet_history_mask_np[idx],
            agent_future_mask_np[idx],
            agent_last_pos_np[idx],
            color_map=_COLOR_MAP[agent_score_type],
            label=agent_label,
            current_only=current_only,
            agent_type=agent_type,
        )

    # plot predictions
    if preds_np.ndim == 4:
        # [n, k, t, 2]
        for idx in range(num_agents):
            agent_score_type = _ID_TO_SCORE_TYPE[int(score_types[idx])]

            if agent_score_type not in _SCORE_TYPES:
                continue
            ax = _plot_predictions(
                ax,
                preds_np[idx],
                probs_np[idx] if probs_np is not None else None,
                max_k=k,
                color_map=_COLOR_MAP[agent_score_type],
                plot_text=(agent_score_type in ["focal", "av"]),
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

    # ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def _plot_lanes(ax: plt.Axes, lane_points_np: np.ndarray, log_id: str = None):
    if log_id is not None:
        visualizer = AV2MapVisualizer()
        visualizer.show_map_clean(ax, seq_id=log_id, show_freespace=False)
        return ax

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
    color_map: Optional[dict] = _COLOR_MAP["unscore"],
    label: Optional[str] = None,
    current_only: bool = True,
    agent_type: str = "default",
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

    # TODO: heading angle
    # if agent_type in _AGENT_BBOX:
    #     bbox_len, bbox_wid = _AGENT_BBOX[agent_type]
    #     ax.add_patch(plt.Rectangle(
    #         (agent_last_pos_np[0] - bbox_len / 2, agent_last_pos_np[1] - bbox_wid / 2),
    #         bbox_len,
    #         bbox_wid,
    #         angle=0.0,
    #         edgecolor=color_map[2],
    #         facecolor='none',
    #         linewidth=0.5,
    #         zorder=4,
    #     ))

    return ax


def _plot_predictions(
    ax: plt.Axes,
    preds_np: np.ndarray,
    probs_np: np.ndarray,
    max_k: int | None = None,
    color_map: Optional[dict] = _COLOR_MAP["unscore"],
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
