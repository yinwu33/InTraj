"""AI trajectory-prediction server for the simax external driver model protocol.

This module implements the same REST API as the existing
``driver_model_server`` (``DummyServer`` / ``SimpleIDMServer``) but predicts
trajectories using a SIMPL deep-learning model.

Usage (from the InTraj directory)::

    uv run python -m serving.ai_model_server [--checkpoint PATH]

The server listens on ``127.0.0.1:1337`` which is the address hard-coded in
the simax C++ ``DriverModelClientAction``.
"""

from __future__ import annotations

import argparse
import math
from enum import Enum
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from datasets.simax.converter import SimaxSimplConverter, SimaxVehicleSnapshot
from simulation.simax.inference import SimplInferenceEngine
from simulation.simax.types import (
    MapInitRequest, ObjectCategory, ObjectState, Trajectory, TrajectoryPoint,
    TrajectoryWithId, VehicleState, VehicleStates, Vector,
)


# ── helpers ───────────────────────────────────────────────────────────

def _to_snapshots(states: VehicleStates) -> List[SimaxVehicleSnapshot]:
    """Convert Pydantic vehicle states to converter snapshots."""
    return [
        SimaxVehicleSnapshot(
            object_id=s.object_id,
            x=s.position.x,
            y=s.position.y,
            z=s.position.z,
            heading=s.heading,
            velocity=s.velocity,
            length=s.length,
            width=s.width,
            category=s.object_category.value,
        )
        for s in states.states
    ]


# ── server ────────────────────────────────────────────────────────────

TIME_INCREMENT_MS = 500  # time between trajectory points (same as IDM server)
DEFAULT_HORIZON = 10  # number of trajectory points to return


class AIModelServer:
    """FastAPI server that predicts trajectories with SIMPL."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.app = FastAPI(title="AI Model Driver Server")

        # Core components.
        self.converter = SimaxSimplConverter()
        self.engine = SimplInferenceEngine(checkpoint_path=checkpoint_path)

        # State.
        self.driver_model_vehicles: dict[int, VehicleState] = {}
        self.non_driver_model_vehicles: dict[int, VehicleState] = {}
        self.current_timestamp: Optional[int] = None
        self.init_timestamp: Optional[int] = None

        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/")
        def redirect_to_docs():
            return RedirectResponse(url="/docs")

        @self.app.post("/initialize_map")
        def post_initialize_map(request: MapInitRequest) -> None:
            print(f"[server] initialize_map: {request.map_file_path}")
            self.converter.set_map(request.map_file_path)

        @self.app.post("/initialize_driver_model_vehicles")
        def post_init_dm(states: VehicleStates) -> None:
            print(f"[server] initialize_driver_model_vehicles: {len(states.states)} vehicles, t={states.time}")
            self.driver_model_vehicles = {s.object_id: s for s in states.states}
            self.init_timestamp = states.time
            self.current_timestamp = states.time
            self.converter.set_driver_model_vehicles(_to_snapshots(states))
            # Record the initial states as the first history step.
            self.converter.update(states.time, _to_snapshots(states))

        @self.app.post("/initialize_non_driver_model_vehicles")
        def post_init_non_dm(states: VehicleStates) -> None:
            print(f"[server] initialize_non_driver_model_vehicles: {len(states.states)} vehicles, t={states.time}")
            self.init_timestamp = states.time
            self.current_timestamp = states.time
            self.non_driver_model_vehicles = {s.object_id: s for s in states.states}
            self.converter.set_non_driver_model_vehicles(_to_snapshots(states))
            # Record the initial states as the first history step.
            self.converter.update(states.time, _to_snapshots(states))

        @self.app.post("/set_states_of_non_driver_model_vehicles")
        def post_set_states(states: VehicleStates) -> None:
            self.current_timestamp = states.time
            self.non_driver_model_vehicles = {s.object_id: s for s in states.states}
            # Also update driver-model vehicles' states via the combined list
            # (the converter accumulates all vehicles for context).
            self.converter.update(states.time, _to_snapshots(states))

        @self.app.get("/get_trajectories")
        def get_trajectories() -> List[TrajectoryWithId]:
            return self._predict_trajectories()

    # ------------------------------------------------------------------

    def _predict_trajectories(self) -> List[TrajectoryWithId]:
        """Run the SIMPL model and convert output to REST format."""
        target_ids = list(self.driver_model_vehicles.keys())
        if not target_ids:
            return []

        # Build batch and run inference.
        batch = self.converter.build_batch(target_vehicle_ids=target_ids)

        # ── DEBUG: model input ──
        ah = batch["agent_history"]       # [B, N_a, T_eff, 14]
        ahm = batch["agent_history_mask"] # [B, N_a, T_eff]
        alp = batch["agent_last_pos"]     # [B, N_a, 2]
        print(f"[DEBUG] batch shapes: agent_history={ah.shape}, "
              f"lane_feats={batch['lane_feats'].shape}, "
              f"rpe[0]={batch['rpe'][0].shape}")
        print(f"[DEBUG] target_ids={target_ids}")
        for b in range(ah.shape[0]):
            valid_steps = int(ahm[b, 0].sum())
            print(f"[DEBUG] target[{b}] id={batch['_target_vehicle_ids'][b]}: "
                  f"last_pos={alp[b, 0].tolist()}, "
                  f"valid_steps={valid_steps}/{ah.shape[2]}")
            disp = ah[b, 0, :, 0:2]  # displacement [T_eff, 2]
            vel = ah[b, 0, :, 4:6]   # velocity [T_eff, 2]
            print(f"[DEBUG]   displacement first5={disp[:5].tolist()}, last5={disp[-5:].tolist()}")
            print(f"[DEBUG]   velocity     first5={vel[:5].tolist()}, last5={vel[-5:].tolist()}")

        pred_positions = self.engine.predict(batch)
        # pred_positions: [B, future_steps, 2] — B = len(target_ids), global coords

        # ── DEBUG: model output ──
        print(f"[DEBUG] pred_positions shape={pred_positions.shape}")
        for b in range(pred_positions.shape[0]):
            vid = batch["_target_vehicle_ids"][b]
            p0 = pred_positions[b, 0]
            p_end = pred_positions[b, -1]
            total_disp = float(np.linalg.norm(p_end - p0))
            print(f"[DEBUG] target[{b}] id={vid}: "
                  f"pred_start={p0.tolist()}, pred_end={p_end.tolist()}, "
                  f"total_displacement={total_disp:.6f}")

        result: List[TrajectoryWithId] = []
        batch_target_ids: List[int] = batch["_target_vehicle_ids"]

        for i, vid in enumerate(batch_target_ids):
            traj_xy = pred_positions[i]  # [future_steps, 2]
            traj = self._positions_to_trajectory(vid, traj_xy)
            result.append(TrajectoryWithId(object_id=vid, trajectory=traj))

        # Self-track DM vehicles: record predicted first-step position as
        # their new state so the next build_batch() anchors correctly.
        # (simax never sends back DM vehicle positions.)
        self._update_dm_positions(batch_target_ids, pred_positions)

        return result

    def _update_dm_positions(
        self, target_ids: List[int], pred_positions: np.ndarray,
    ) -> None:
        """Advance DM vehicles by one step using constant velocity.

        simax never sends back DM vehicle positions, so the converter
        would be stuck at the init position forever.  We advance them
        using their *original* heading and velocity (from the init
        endpoint) to keep the history smooth and directionally correct.
        Using predictions for heading causes noisy feedback loops.
        """
        DT = 0.1  # must match converter back-fill DT (10 Hz)
        for i, vid in enumerate(target_ids):
            dm_state = self.driver_model_vehicles.get(vid)
            if dm_state is None:
                continue

            heading = dm_state.heading
            velocity = dm_state.velocity

            # Current position: latest snapshot in converter history.
            buf = self.converter._history.get(vid)
            if buf:
                last_snap = buf[-1]
                cur_x, cur_y = last_snap.x, last_snap.y
            else:
                cur_x, cur_y = dm_state.position.x, dm_state.position.y

            new_x = cur_x + velocity * math.cos(heading) * DT
            new_y = cur_y + velocity * math.sin(heading) * DT

            snap = SimaxVehicleSnapshot(
                object_id=vid,
                x=new_x, y=new_y, z=0.0,
                heading=heading,
                velocity=velocity,
                length=dm_state.length, width=dm_state.width,
                category=dm_state.object_category.value,
            )
            self.converter._history[vid].append(snap)

    def _positions_to_trajectory(
        self, vehicle_id: int, positions: np.ndarray,
    ) -> Trajectory:
        """Convert an ``[T, 2]`` position array to a :class:`Trajectory`.

        Generates ``DEFAULT_HORIZON + 2`` points (matching the IDM server)
        by sub-sampling or repeating the model output as needed.
        """
        n_pts = DEFAULT_HORIZON + 2
        total_predicted = positions.shape[0]

        # Sub-sample if we have more predictions than needed.
        if total_predicted >= n_pts:
            indices = np.linspace(0, total_predicted - 1, n_pts, dtype=int)
            sampled = positions[indices]
        else:
            # Pad with last position.
            pad = np.repeat(positions[-1:], n_pts - total_predicted, axis=0)
            sampled = np.concatenate([positions, pad], axis=0)

        t_ms = self.current_timestamp if self.current_timestamp is not None else 0

        # Compute headings from displacement.
        points: List[TrajectoryPoint] = []
        for j in range(n_pts):
            x, y = float(sampled[j, 0]), float(sampled[j, 1])
            if j < n_pts - 1:
                dx = float(sampled[j + 1, 0] - sampled[j, 0])
                dy = float(sampled[j + 1, 1] - sampled[j, 1])
                heading = math.atan2(dy, dx)
            else:
                heading = points[-1].heading if points else 0.0

            points.append(
                TrajectoryPoint(
                    time=t_ms,
                    position=Vector(x=x, y=y, z=0.0),
                    heading=heading,
                )
            )
            t_ms += TIME_INCREMENT_MS

        return Trajectory(points=points)


# ── entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Model Driver Server")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a SIMPL checkpoint (.ckpt or .pt)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
    )
    parser.add_argument(
        "--port", type=int, default=1337,
    )
    args = parser.parse_args()

    server = AIModelServer(checkpoint_path=args.checkpoint)
    print(f"[server] Starting AI Model Server on {args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
