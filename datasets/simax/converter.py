"""Convert simax simulation data to SIMPL-compatible batch format.

This module provides :class:`SimaxSimplConverter` which accumulates
vehicle states received from the simax REST protocol and converts them into
the ``dict`` expected by :class:`models.simpl.simpl.Simpl`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lightweight OSM / lanelet2 map parser (no lanelet2 Python binding needed)
# ---------------------------------------------------------------------------

def _parse_lanelet_map_osm(map_file_path: str) -> list[np.ndarray]:
    """Parse a lanelet2 ``.osm`` file and return lane centerlines.

    Returns a list of ``[N_points, 2]`` arrays (x, y) in local metres.
    WGS84 lat/lon are projected to a local tangent plane via an
    equirectangular approximation centred on the map-node centroid.
    """
    try:
        tree = ET.parse(map_file_path)
    except Exception as exc:
        print(f"[simax converter] WARNING: failed to parse map {map_file_path}: {exc}")
        return []

    root = tree.getroot()

    # Collect raw WGS84 coordinates.
    nodes_wgs: Dict[int, Tuple[float, float]] = {}
    for node_el in root.iter("node"):
        nid = int(node_el.get("id", 0))
        lat = float(node_el.get("lat", 0.0))
        lon = float(node_el.get("lon", 0.0))
        nodes_wgs[nid] = (lat, lon)

    if not nodes_wgs:
        return []

    # Reference point = centroid of all map nodes.
    all_ll = np.array(list(nodes_wgs.values()), dtype=np.float64)
    lat_ref = float(all_ll[:, 0].mean())
    lon_ref = float(all_ll[:, 1].mean())
    cos_lat = np.cos(np.radians(lat_ref))

    # Equirectangular projection → local (x, y) metres.
    nodes: Dict[int, Tuple[float, float]] = {}
    for nid, (lat, lon) in nodes_wgs.items():
        x = (lon - lon_ref) * cos_lat * 111319.5
        y = (lat - lat_ref) * 111319.5
        nodes[nid] = (x, y)

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    print(
        f"[simax converter] Map projection ref=({lat_ref:.8f}, {lon_ref:.8f})"
        f"  range x=[{min(xs):.2f}, {max(xs):.2f}] y=[{min(ys):.2f}, {max(ys):.2f}]"
    )

    ways: Dict[int, List[Tuple[float, float]]] = {}
    for way_el in root.iter("way"):
        wid = int(way_el.get("id", 0))
        pts = []
        for nd_el in way_el.iter("nd"):
            ref = int(nd_el.get("ref", 0))
            if ref in nodes:
                pts.append(nodes[ref])
        if pts:
            ways[wid] = pts

    centerlines: list[np.ndarray] = []
    for rel_el in root.iter("relation"):
        is_lanelet = False
        for tag_el in rel_el.iter("tag"):
            if tag_el.get("k") == "type" and tag_el.get("v") == "lanelet":
                is_lanelet = True
                break
        if not is_lanelet:
            continue

        left_id: Optional[int] = None
        right_id: Optional[int] = None
        for member_el in rel_el.iter("member"):
            role = member_el.get("role", "")
            if member_el.get("type") == "way":
                ref = int(member_el.get("ref", 0))
                if role == "left":
                    left_id = ref
                elif role == "right":
                    right_id = ref

        if left_id is not None and right_id is not None:
            left_pts = ways.get(left_id)
            right_pts = ways.get(right_id)
            if left_pts and right_pts:
                left_arr = np.array(left_pts, dtype=np.float64)
                right_arr = np.array(right_pts, dtype=np.float64)
                n_pts = max(len(left_arr), len(right_arr))
                if len(left_arr) != n_pts:
                    left_arr = _interp_polyline(left_arr, n_pts)
                if len(right_arr) != n_pts:
                    right_arr = _interp_polyline(right_arr, n_pts)
                center = ((left_arr + right_arr) / 2.0).astype(np.float32)
                centerlines.append(center)

    return centerlines


def _interp_polyline(pts: np.ndarray, target_n: int) -> np.ndarray:
    """Linearly interpolate a polyline to *target_n* points."""
    if pts.shape[0] < 2 or target_n < 2:
        return np.repeat(pts[:1], target_n, axis=0)
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.insert(np.cumsum(dists), 0, 0.0)
    targets = np.linspace(0.0, cum[-1], target_n)
    result = np.zeros((target_n, pts.shape[1]), dtype=pts.dtype)
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum, t)
        if idx == 0:
            result[i] = pts[0]
        elif idx >= len(cum):
            result[i] = pts[-1]
        else:
            frac = (t - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
            result[i] = pts[idx - 1] + frac * (pts[idx] - pts[idx - 1])
    return result


def _resample_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """Uniformly resample a polyline to *num_points* points (xy only)."""
    pts = polyline[:, :2] if polyline.shape[1] > 2 else polyline
    if pts.shape[0] == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, num_points, axis=0).astype(np.float32)
    return _interp_polyline(pts, num_points).astype(np.float32)


def _rotation_matrix(heading: float) -> np.ndarray:
    """2×2 rotation matrix for *heading* (radians)."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Vehicle-state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimaxVehicleSnapshot:
    """One vehicle at one time-step."""
    object_id: int
    x: float
    y: float
    z: float
    heading: float
    velocity: float
    length: float
    width: float
    category: str


# ---------------------------------------------------------------------------
# Converter  (SIMPL)
# ---------------------------------------------------------------------------

class SimaxSimplConverter:
    """Accumulates simax state updates and builds SIMPL batch dicts.

    Parameters match the SIMPL dataset defaults (see
    ``configs/datamodule/motion_av2_simpl.yaml``).
    """

    def __init__(
        self,
        history_steps: int = 50,
        truncate_steps: int = 2,
        future_steps: int = 60,
        max_agents: int = 64,
        max_lanes: int = 256,
        points_per_polyline: int = 11,
        rpe_radius: float = 150.0,
    ) -> None:
        self.history_steps = history_steps
        self.truncate_steps = truncate_steps
        self.effective_T = history_steps - truncate_steps  # 48
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.points_per_polyline = points_per_polyline
        self.num_lane_nodes = points_per_polyline - 1  # 10 segments
        self.rpe_radius = rpe_radius

        # Cached lane data — set once via set_map().
        self._lane_centerlines: list[np.ndarray] = []  # raw [N_pts, 2]

        # Per-vehicle history buffers.
        self._history: Dict[int, deque[SimaxVehicleSnapshot]] = {}
        self._timestamps_ms: deque[int] = deque(maxlen=history_steps)

        # Role bookkeeping.
        self._driver_model_ids: set[int] = set()
        self._ego_id: Optional[int] = None

    # -- map ----------------------------------------------------------------

    def set_map(self, map_file_path: str) -> None:
        raw_centerlines = _parse_lanelet_map_osm(map_file_path)
        if not raw_centerlines:
            print("[simax converter] No lanes parsed — model will run without map context.")
            self._lane_centerlines = []
            return
        self._lane_centerlines = raw_centerlines
        print(f"[simax converter] Parsed {len(self._lane_centerlines)} lane(s) from map.")

    # -- vehicle bookkeeping ------------------------------------------------

    def set_driver_model_vehicles(self, snapshots: List[SimaxVehicleSnapshot]) -> None:
        self._driver_model_ids = {v.object_id for v in snapshots}
        self._record_snapshots(snapshots)

    def set_non_driver_model_vehicles(self, snapshots: List[SimaxVehicleSnapshot]) -> None:
        for v in snapshots:
            if v.object_id == 0:
                self._ego_id = v.object_id
        self._record_snapshots(snapshots)

    # -- tick ---------------------------------------------------------------

    def update(self, time_ms: int, snapshots: List[SimaxVehicleSnapshot]) -> None:
        self._timestamps_ms.append(time_ms)
        self._record_snapshots(snapshots)

    # -- batch building (SIMPL format) -------------------------------------

    def build_batch(self, target_vehicle_ids: Sequence[int]) -> dict:
        """Build a SIMPL-compatible batch dict for inference.

        SIMPL expects batched, padded tensors with B=1 for online inference.
        Each target vehicle becomes a separate batch item where the target
        is placed at agent index 0.
        """
        n_steps = len(self._timestamps_ms)
        if n_steps == 0:
            raise RuntimeError("No states recorded yet")

        target_ids = [vid for vid in target_vehicle_ids if vid in self._history]
        if not target_ids:
            raise RuntimeError("No target vehicles in history")

        # Collect global positions, headings, velocities for all agents.
        all_ids = list(self._history.keys())
        T = self.history_steps
        A = len(all_ids)

        positions = np.zeros((A, T, 2), dtype=np.float32)
        headings = np.zeros((A, T), dtype=np.float32)
        velocities = np.zeros((A, T, 2), dtype=np.float32)
        valid = np.zeros((A, T), dtype=bool)

        for ai, vid in enumerate(all_ids):
            buf = self._history[vid]
            offset = T - min(len(buf), T)
            for bi, snap in enumerate(buf):
                if bi >= T:
                    break
                idx = offset + bi
                positions[ai, idx] = [snap.x, snap.y]
                headings[ai, idx] = snap.heading
                vx = snap.velocity * np.cos(snap.heading)
                vy = snap.velocity * np.sin(snap.heading)
                velocities[ai, idx] = [vx, vy]
                valid[ai, idx] = True

        # Back-fill missing early history by extrapolating from the
        # earliest known state using its velocity and heading.  This gives
        # the model a plausible constant-velocity history instead of a
        # cluster of identical positions (which produces zero displacement
        # and makes the model predict "stay still").
        #
        # We also forward-fill any remaining gaps (shouldn't happen in
        # practice but keeps the code defensive).
        #
        # Use fixed DT = 0.1 s (10 Hz) to match the AV2 training data
        # frame rate.  Deriving DT from timestamps is unreliable because
        # both init endpoints may share the same timestamp.
        DT = 0.1
        for ai in range(A):
            # --- back-fill: find first valid index, extrapolate backwards ---
            first_valid = -1
            for t in range(T):
                if valid[ai, t]:
                    first_valid = t
                    break
            if first_valid > 0:
                ref_pos = positions[ai, first_valid]
                ref_heading = headings[ai, first_valid]
                ref_vel = velocities[ai, first_valid]
                speed = np.linalg.norm(ref_vel)
                dx = speed * np.cos(ref_heading) * DT
                dy = speed * np.sin(ref_heading) * DT
                for t in range(first_valid - 1, -1, -1):
                    steps_back = first_valid - t
                    positions[ai, t] = ref_pos - steps_back * np.array([dx, dy], dtype=np.float32)
                    headings[ai, t] = ref_heading
                    velocities[ai, t] = ref_vel
                    valid[ai, t] = True

            # --- forward-fill any remaining gaps ---
            for t in range(1, T):
                if not valid[ai, t]:
                    positions[ai, t] = positions[ai, t - 1]
                    headings[ai, t] = headings[ai, t - 1]
                    velocities[ai, t] = velocities[ai, t - 1]

        # Build lane features in global frame.
        lane_pts_global, lane_ctrs, lane_vecs, lane_masks_np = self._build_lane_features_global()

        # Build one batch item per target vehicle.
        batch_agent_history = []
        batch_agent_history_mask = []
        batch_agent_last_pos = []
        batch_agent_last_rot = []
        batch_lane_feats = []
        batch_lane_masks = []
        batch_rpe = []

        for tid in target_ids:
            t_idx = all_ids.index(tid)

            # Order agents: target first, then rest (up to max_agents).
            agent_order = [t_idx] + [i for i in range(A) if i != t_idx]
            agent_order = agent_order[: self.max_agents]
            N_a = len(agent_order)

            # Per-agent local frame: based on each agent's last position/heading.
            last_pos_agents = np.zeros((N_a, 2), dtype=np.float32)
            last_heading_agents = np.zeros(N_a, dtype=np.float32)
            last_rot_agents = np.zeros((N_a, 2, 2), dtype=np.float32)

            for j, ai in enumerate(agent_order):
                last_pos_agents[j] = positions[ai, T - 1]
                last_heading_agents[j] = headings[ai, T - 1]
                last_rot_agents[j] = _rotation_matrix(headings[ai, T - 1])

            # Build 14-dim agent history features in each agent's local frame.
            hist_feat = np.zeros((N_a, T, 14), dtype=np.float32)
            hist_mask = np.zeros((N_a, T), dtype=bool)

            for j, ai in enumerate(agent_order):
                rot = last_rot_agents[j]  # [2, 2]
                pos_local = (positions[ai] - last_pos_agents[j]) @ rot  # [T, 2]
                heading_local = _wrap_angle(headings[ai] - last_heading_agents[j])  # [T]
                vel_local = velocities[ai] @ rot  # [T, 2]

                # Displacement (step-to-step).
                displacement = np.zeros((T, 2), dtype=np.float32)
                displacement[1:] = pos_local[1:] - pos_local[:-1]

                # Agent type one-hot (7 dims): index 0 = vehicle (default for simax).
                agent_onehot = np.zeros(7, dtype=np.float32)
                agent_onehot[0] = 1.0  # vehicle

                hist_feat[j, :, 0:2] = displacement           # dx, dy
                hist_feat[j, :, 2] = np.cos(heading_local)     # cos(heading)
                hist_feat[j, :, 3] = np.sin(heading_local)     # sin(heading)
                hist_feat[j, :, 4:6] = vel_local               # vx, vy
                hist_feat[j, :, 6:13] = agent_onehot           # one-hot
                hist_feat[j, :, 13] = valid[ai].astype(np.float32)  # validity
                hist_mask[j] = valid[ai]

            # Truncate first `truncate_steps` steps.
            hist_feat = hist_feat[:, self.truncate_steps:, :]    # [N_a, T_eff, 14]
            hist_mask = hist_mask[:, self.truncate_steps:]       # [N_a, T_eff]

            # Build lane features in polyline-local frame.
            N_l = lane_ctrs.shape[0] if lane_ctrs.shape[0] > 0 else 0
            lane_feats_local = np.zeros(
                (N_l, self.num_lane_nodes, 16), dtype=np.float32,
            )
            if N_l > 0:
                for li in range(N_l):
                    pts = lane_pts_global[li]  # [points_per_polyline, 2]
                    ctr = lane_ctrs[li]
                    vec = lane_vecs[li]
                    lane_heading = np.arctan2(vec[1], vec[0])
                    lane_rot = _rotation_matrix(lane_heading)

                    pts_local = (pts - ctr) @ lane_rot  # [P, 2]
                    # Node features from consecutive point pairs.
                    node_ctrs_l = (pts_local[:-1] + pts_local[1:]) / 2.0  # [N_nodes, 2]
                    node_vecs_l = pts_local[1:] - pts_local[:-1]          # [N_nodes, 2]

                    lane_feats_local[li, :, 0:2] = node_ctrs_l        # segment midpoint
                    lane_feats_local[li, :, 2:4] = node_vecs_l        # segment vector
                    # channels 4 (intersection), 5-11 (map type one-hot), 12-15 (padding) stay 0

            # Build RPE [5, N_a + N_l, N_a + N_l].
            rpe = self._build_rpe(
                last_pos_agents, last_heading_agents,
                lane_ctrs, lane_vecs,
            )

            # Pad to max_agents / max_lanes.
            N_a_pad = max(N_a, 1)
            N_l_pad = max(N_l, 1)

            ah = np.zeros((N_a_pad, self.effective_T, 14), dtype=np.float32)
            ah[:N_a] = hist_feat
            ahm = np.zeros((N_a_pad, self.effective_T), dtype=bool)
            ahm[:N_a] = hist_mask
            alp = np.zeros((N_a_pad, 2), dtype=np.float32)
            alp[:N_a] = last_pos_agents
            alr = np.zeros((N_a_pad, 2, 2), dtype=np.float32)
            alr[:N_a] = last_rot_agents

            lf = np.zeros((N_l_pad, self.num_lane_nodes, 16), dtype=np.float32)
            if N_l > 0:
                lf[:N_l] = lane_feats_local
            lm = np.zeros(N_l_pad, dtype=bool)
            if N_l > 0:
                lm[:N_l] = lane_masks_np[:N_l]

            batch_agent_history.append(torch.from_numpy(ah))
            batch_agent_history_mask.append(torch.from_numpy(ahm))
            batch_agent_last_pos.append(torch.from_numpy(alp))
            batch_agent_last_rot.append(torch.from_numpy(alr))
            batch_lane_feats.append(torch.from_numpy(lf))
            batch_lane_masks.append(torch.from_numpy(lm))
            batch_rpe.append(torch.from_numpy(rpe).float())

        # Stack into batch (B = len(target_ids)).
        B = len(target_ids)
        # Pad all items to the same N_a_max / N_l_max.
        N_a_max = max(t.shape[0] for t in batch_agent_history)
        N_l_max = max(t.shape[0] for t in batch_lane_feats)

        agent_history_t = torch.zeros((B, N_a_max, self.effective_T, 14))
        agent_history_mask_t = torch.zeros((B, N_a_max, self.effective_T), dtype=torch.bool)
        agent_last_pos_t = torch.zeros((B, N_a_max, 2))
        agent_last_rot_t = torch.zeros((B, N_a_max, 2, 2))
        lane_feats_t = torch.zeros((B, N_l_max, self.num_lane_nodes, 16))
        lane_masks_t = torch.zeros((B, N_l_max), dtype=torch.bool)

        for i in range(B):
            na = batch_agent_history[i].shape[0]
            nl = batch_lane_feats[i].shape[0]
            agent_history_t[i, :na] = batch_agent_history[i]
            agent_history_mask_t[i, :na] = batch_agent_history_mask[i]
            agent_last_pos_t[i, :na] = batch_agent_last_pos[i]
            agent_last_rot_t[i, :na] = batch_agent_last_rot[i]
            lane_feats_t[i, :nl] = batch_lane_feats[i]
            lane_masks_t[i, :nl] = batch_lane_masks[i]

        return {
            "agent_history": agent_history_t,
            "agent_history_mask": agent_history_mask_t,
            "agent_last_pos": agent_last_pos_t,
            "agent_last_rot": agent_last_rot_t,
            "lane_feats": lane_feats_t,
            "lane_masks": lane_masks_t,
            "rpe": batch_rpe,
            # metadata
            "_target_vehicle_ids": list(target_ids),
        }

    # -- internal helpers ---------------------------------------------------

    def _record_snapshots(self, snapshots: List[SimaxVehicleSnapshot]) -> None:
        for s in snapshots:
            buf = self._history.setdefault(
                s.object_id, deque(maxlen=self.history_steps),
            )
            buf.append(s)

    def _build_lane_features_global(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resample lanes and compute per-lane center / direction vectors.

        Returns (lane_pts, lane_ctrs, lane_vecs, lane_masks).
        """
        P = self.points_per_polyline
        centerlines = self._lane_centerlines
        if not centerlines:
            empty2 = np.zeros((0, 2), dtype=np.float32)
            return (
                np.zeros((0, P, 2), dtype=np.float32),
                empty2, empty2,
                np.zeros(0, dtype=bool),
            )

        # Limit lanes.
        if len(centerlines) > self.max_lanes:
            dists = [np.linalg.norm(cl.mean(axis=0)) for cl in centerlines]
            order = np.argsort(dists)[: self.max_lanes]
            centerlines = [centerlines[i] for i in order]

        N_l = len(centerlines)
        lane_pts = np.zeros((N_l, P, 2), dtype=np.float32)
        lane_ctrs = np.zeros((N_l, 2), dtype=np.float32)
        lane_vecs = np.zeros((N_l, 2), dtype=np.float32)
        lane_masks = np.ones(N_l, dtype=bool)

        for li, cl in enumerate(centerlines):
            resampled = _resample_polyline(cl, P)
            lane_pts[li] = resampled
            lane_ctrs[li] = resampled.mean(axis=0)
            direction = resampled[-1] - resampled[0]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                lane_vecs[li] = direction / norm
            else:
                lane_vecs[li] = [1.0, 0.0]

        return lane_pts, lane_ctrs, lane_vecs, lane_masks

    def _build_rpe(
        self,
        agent_pos: np.ndarray,     # [N_a, 2]
        agent_heading: np.ndarray,  # [N_a]
        lane_ctrs: np.ndarray,      # [N_l, 2]
        lane_vecs: np.ndarray,      # [N_l, 2]
    ) -> np.ndarray:
        """Build the Relative Position Encoding matrix [5, N, N]."""
        N_a = agent_pos.shape[0]
        N_l = lane_ctrs.shape[0]
        N = N_a + N_l

        if N == 0:
            return np.zeros((5, 1, 1), dtype=np.float32)

        # Scene points and unit heading vectors.
        scene_pts = np.zeros((N, 2), dtype=np.float32)
        scene_vecs = np.zeros((N, 2), dtype=np.float32)

        scene_pts[:N_a] = agent_pos
        for i in range(N_a):
            scene_vecs[i] = [np.cos(agent_heading[i]), np.sin(agent_heading[i])]

        if N_l > 0:
            scene_pts[N_a:] = lane_ctrs
            scene_vecs[N_a:] = lane_vecs

        # Pairwise displacement.
        diff = scene_pts[None, :, :] - scene_pts[:, None, :]  # [N, N, 2]
        dist = np.linalg.norm(diff, axis=-1)                  # [N, N]

        # Helpers for cos/sin between vectors.
        def _cos_sin(a: np.ndarray, b: np.ndarray):
            # a, b: [N, N, 2]
            dot = (a * b).sum(axis=-1)
            cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            norm_a = np.linalg.norm(a, axis=-1).clip(1e-8)
            norm_b = np.linalg.norm(b, axis=-1).clip(1e-8)
            denom = norm_a * norm_b
            return dot / denom, cross / denom

        vec_i = np.broadcast_to(scene_vecs[:, None, :], (N, N, 2))
        vec_j = np.broadcast_to(scene_vecs[None, :, :], (N, N, 2))

        cos_vv, sin_vv = _cos_sin(vec_i, vec_j)
        cos_vd, sin_vd = _cos_sin(vec_i, diff)

        rpe = np.stack([
            cos_vv,                              # [N, N]
            sin_vv,                              # [N, N]
            cos_vd,                              # [N, N]
            sin_vd,                              # [N, N]
            dist / (self.rpe_radius * 2.0),      # [N, N]
        ], axis=0)  # [5, N, N]

        return rpe.astype(np.float32)
