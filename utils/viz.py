from __future__ import annotations
from typing import Optional

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe headless backend for dataloader forks
import matplotlib.pyplot as plt


def _to_numpy(arr: torch.Tensor | np.ndarray | None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def plot_scenario(
    lane_points: torch.Tensor | np.ndarray,
    agent_history: torch.Tensor | np.ndarray,
    target_agent_idx: int,
    target_last_pos: torch.Tensor | np.ndarray | None = None,
    target_gt: torch.Tensor | np.ndarray | None = None,
    prediction: torch.Tensor | np.ndarray | None = None,
    other_future: torch.Tensor | np.ndarray | None = None,
    other_future_mask: torch.Tensor | np.ndarray | None = None,
    other_prediction: torch.Tensor | np.ndarray | None = None,
    agent_last_pos: torch.Tensor | np.ndarray | None = None,
    scenario_id: str | None = None,
    view_radius: float = 80.0,
):
    """Plot lanes, agent history, target future ground truth, and prediction."""

    lane_points_np = _to_numpy(lane_points)
    agent_history_np = _to_numpy(agent_history)
    target_last_pos_np = _to_numpy(target_last_pos)
    target_gt_np = _to_numpy(target_gt)
    pred_np = _to_numpy(prediction)
    other_future_np = _to_numpy(other_future)
    other_future_mask_np = _to_numpy(other_future_mask)
    other_pred_np = _to_numpy(other_prediction)
    agent_last_np = _to_numpy(agent_last_pos)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Map polylines
    for lane in lane_points_np:
        ax.plot(lane[:, 0], lane[:, 1], color="#c7c7c7",
                linewidth=1.0, zorder=0)

    # Agent histories
    obs_mask = agent_history_np[:, :, 6] > 0.5
    for idx, seq in enumerate(agent_history_np):
        mask = obs_mask[idx]
        if not mask.any():
            continue
        coords = seq[mask, :2]
        if idx == target_agent_idx:
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="#1f77b4",
                linewidth=2.0,
                label="target history",
            )
            ax.scatter(
                coords[-1, 0],
                coords[-1, 1],
                color="#1f77b4",
                s=20,
                zorder=5,
            )
        else:
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="#7f7f7f",
                linewidth=1.0,
                alpha=0.7,
            )
            ax.scatter(
                coords[-1, 0],
                coords[-1, 1],
                color="#7f7f7f",
                s=8,
                alpha=0.8,
                zorder=4,
            )

    # Future ground truth
    if target_gt_np is not None:
        gt_coords = target_gt_np
        if target_last_pos_np is not None:
            gt_coords = np.vstack([target_last_pos_np[None, :], gt_coords])
        ax.plot(
            gt_coords[:, 0],
            gt_coords[:, 1],
            color="#2ca02c",
            linewidth=2.0,
            label="target future",
        )
        ax.scatter(gt_coords[-1, 0], gt_coords[-1, 1], color="#2ca02c", s=25)

    # Model prediction
    if pred_np is not None:
        target_pred = None
        if pred_np.ndim == 4 and pred_np.shape[0] > target_agent_idx:
            target_pred = pred_np[target_agent_idx][0]  # take first mode
        elif pred_np.ndim == 3 and pred_np.shape[0] > target_agent_idx:
            target_pred = (
                pred_np[target_agent_idx]
                if pred_np.shape[1] == 2
                else pred_np[target_agent_idx][0]
            )
        else:
            target_pred = pred_np

        if target_pred is not None and target_pred.ndim == 2:
            pred_coords = target_pred
            if target_last_pos_np is not None:
                pred_coords = np.vstack(
                    [target_last_pos_np[None, :], pred_coords])
            ax.plot(
                pred_coords[:, 0],
                pred_coords[:, 1],
                color="#d62728",
                linestyle="--",
                linewidth=2.0,
                label="prediction",
            )
            ax.scatter(pred_coords[-1, 0],
                       pred_coords[-1, 1], color="#d62728", s=25)

    # Other agents future GT (lighter)
    if other_future_np is not None and other_future_np.ndim == 3:
        plotted = False
        for idx in range(other_future_np.shape[0]):
            if idx == target_agent_idx:
                continue
            traj = other_future_np[idx]
            if other_future_mask_np is not None and other_future_mask_np.shape[0] > idx:
                mask = other_future_mask_np[idx] > 0.5
                if mask.any():
                    traj = traj[mask]
                else:
                    continue
            if traj.shape[0] == 0:
                continue
            start = (
                agent_last_np[idx]
                if agent_last_np is not None and agent_last_np.shape[0] > idx
                else None
            )
            if start is not None:
                traj = np.vstack([start[None, :], traj])
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="#9ed9a0",
                linewidth=1.2,
                alpha=0.9,
                label="others future" if not plotted else None,
            )
            plotted = True

    # Other agents predictions (lighter)
    if other_pred_np is not None and other_pred_np.ndim >= 3:
        plotted = False
        num_agents = other_pred_np.shape[0]
        for idx in range(num_agents):
            if idx == target_agent_idx:
                continue
            traj = other_pred_np[idx]
            if traj.ndim == 3:  # has mode dimension
                traj = traj[0]
            start = (
                agent_last_np[idx]
                if agent_last_np is not None and agent_last_np.shape[0] > idx
                else None
            )
            if start is not None:
                traj = np.vstack([start[None, :], traj])
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="#f7b6d2",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label="others pred" if not plotted else None,
            )
            plotted = True

    if target_last_pos_np is not None:
        cx, cy = target_last_pos_np
        ax.set_xlim(cx - view_radius, cx + view_radius)
        ax.set_ylim(cy - view_radius, cy + view_radius)

    ax.set_aspect("equal", adjustable="box")
    title = "Scenario visualization"
    if scenario_id is not None:
        title += f" ({scenario_id})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig
