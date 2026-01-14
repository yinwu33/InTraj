from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import pickle
import numpy as np
import torch
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import Dict, Any

# --- AV2 API ---
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    TrackCategory,
    ObjectType,
)

from datamodule.datasets.av2.av2_base_dataset import AV2BaseDataset

# --- Configuration ---
DATA_ROOT = "/home/tjhu78u/workspace/motion_prediction/data"
OUTPUT_ROOT = "/home/tjhu78u/workspace/motion_prediction/cache/av2_infgen"
SPLIT = "train"

# --- Waymo compatibility ---
# Waymo motion preprocessing typically uses 91 steps (11 history incl. current + 80 future)
NUM_STEPS = 91

# --- Mappings to match Waymo Preprocess Enums ---

# Waymo: ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
_POLYGON_TYPES = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
AV2_LANE_TYPE_TO_ID = {
    LaneType.VEHICLE: _POLYGON_TYPES.index('VEHICLE'),
    LaneType.BIKE: _POLYGON_TYPES.index('BIKE'),
    LaneType.BUS: _POLYGON_TYPES.index('BUS'),
}

# Waymo point types
_POINT_TYPES = [
    'DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
    'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
    'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'EDGE',
    'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE'
]
POINT_TYPE_CENTERLINE = _POINT_TYPES.index('CENTERLINE')
POINT_TYPE_CROSSWALK = _POINT_TYPES.index('CROSSWALK')

# Polygon edge types
_POLYGON_TO_POLYGON_TYPES = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']


def get_agent_features_av2(scenario: ArgoverseScenario, num_steps: int = NUM_STEPS) -> Dict[str, Any]:
    """Extract agent features in the same *shape* conventions used by your Waymo preprocessing.

    Changes from your original:
      1) Truncate time to the first `num_steps` (default 91).
      2) Keep `shape` as [N, T, 3] as requested.
      3) Do NOT change your type/id logic (per your request).
    """

    track_ids = [track.track_id for track in scenario.tracks]
    num_agents = len(track_ids)

    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    role = torch.zeros(num_agents, 3, dtype=torch.bool)  # [Ego, Interest, Predict]
    agent_id_list = []
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    category = torch.zeros(num_agents, dtype=torch.uint8)  # 1 if predicted, 0 otherwise

    position = torch.zeros(num_agents, num_steps, 3, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    shape = torch.zeros(num_agents, num_steps, 3, dtype=torch.float)  # L, W, H

    type_map = {
        ObjectType.VEHICLE: 0,
        ObjectType.PEDESTRIAN: 1,
        ObjectType.CYCLIST: 2,
        ObjectType.BUS: 0,
        ObjectType.MOTORCYCLIST: 2,
        ObjectType.UNKNOWN: 0,
        ObjectType.STATIC: 0,
        ObjectType.BACKGROUND: 0,
    }

    av_idx = -1

    for i, track in enumerate(scenario.tracks):
        agent_id_list.append(track.track_id)

        cur_type = type_map.get(track.object_type, 0)

        is_av = (track.track_id == "AV")
        is_focal = (track.track_id == scenario.focal_track_id)
        is_scored = (track.category == TrackCategory.SCORED_TRACK)

        if is_av:
            role[i, 0] = True
            av_idx = i
            if cur_type == 0:
                cur_type = 1

        if is_focal:
            role[i, 1] = True

        if is_scored or is_focal:
            role[i, 2] = True
            category[i] = 1
            if cur_type == 0:
                cur_type = 1

        agent_type[i] = cur_type

        # States: keep only timestep < num_steps
        sorted_states = sorted(track.object_states, key=lambda x: x.timestep)

        for state in sorted_states:
            t_idx = state.timestep
            if 0 <= t_idx < num_steps:
                valid_mask[i, t_idx] = True

                position[i, t_idx, 0] = state.position[0]
                position[i, t_idx, 1] = state.position[1]
                position[i, t_idx, 2] = 0.0  # AV2 mostly 2D

                heading[i, t_idx] = state.heading

                velocity[i, t_idx, 0] = state.velocity[0]
                velocity[i, t_idx, 1] = state.velocity[1]

                # Default dimensions based on type
                l, w, h = 4.5, 2.0, 1.6
                if track.object_type == ObjectType.PEDESTRIAN:
                    l, w, h = 0.6, 0.6, 1.7
                elif track.object_type in [ObjectType.CYCLIST, ObjectType.MOTORCYCLIST]:
                    l, w, h = 2.0, 0.7, 1.5
                elif track.object_type == ObjectType.BUS:
                    l, w, h = 12.0, 2.5, 3.0

                shape[i, t_idx, 0] = l
                shape[i, t_idx, 1] = w
                shape[i, t_idx, 2] = h

    # Keep your original approach: hash string ids to int64 (requested: do not modify)
    id_tensor = torch.tensor([hash(x) % ((1 << 63) - 1) for x in agent_id_list], dtype=torch.int64)
    
    data =  {
        'num_nodes': num_agents,
        'valid_mask': valid_mask,
        'role': role,
        'id': id_tensor,
        'type': agent_type,
        'category': category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
        'shape': shape,
        'av_idx': av_idx,
    }
    
    data = filter_non_show_agent(data)
    
    for i_agent_shape in data["shape"]:
        assert torch.all(i_agent_shape == 0.0) == False, "Found agent with all-zero shape after filtering!"
    
    return data
    
def filter_non_show_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter agents that never appear in the time window (valid_mask is all False).

    Requirements implemented:
      - Use valid_mask to compute valid agent count and indices
      - Create new tensors
      - Copy/merge input data into output, with agent fields filtered

    Expected input format:
      data['agent'] contains:
        - 'valid_mask': [N, T] bool
        - 'role':       [N, 3] bool
        - 'id':         [N] int64
        - 'type':       [N] uint8
        - 'category':   [N] uint8
        - 'position':   [N, T, 3] float
        - 'heading':    [N, T] float
        - 'velocity':   [N, T, 2] float
        - 'shape':      [N, T, 3] float
        - 'av_idx':     int
        - 'num_nodes':  int
    """

    valid_mask: torch.Tensor = data["valid_mask"]  # [N, T]
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask should be [N,T], got shape={tuple(valid_mask.shape)}")

    # 1) valid agent indices
    keep = valid_mask.any(dim=1)  # [N]
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)  # [M]
    num_valid = int(keep_idx.numel())

    # If no valid agents, return empty agent tensors but keep map/scenario keys
    if num_valid == 0:
        out = dict(data)  # shallow copy top-level
        T = valid_mask.shape[1]
        out["agent"] = {
            "num_nodes": 0,
            "valid_mask": torch.zeros((0, T), dtype=torch.bool, device=valid_mask.device),
            "role": torch.zeros((0, 3), dtype=torch.bool, device=valid_mask.device),
            "id": torch.zeros((0,), dtype=data["id"].dtype, device=data["id"].device),
            "type": torch.zeros((0,), dtype=data["type"].dtype, device=data["type"].device),
            "category": torch.zeros((0,), dtype=data["category"].dtype, device=data["category"].device),
            "position": torch.zeros((0, T, 3), dtype=data["position"].dtype, device=data["position"].device),
            "heading": torch.zeros((0, T), dtype=data["heading"].dtype, device=data["heading"].device),
            "velocity": torch.zeros((0, T, 2), dtype=data["velocity"].dtype, device=data["velocity"].device),
            "shape": torch.zeros((0, T, 3), dtype=data["shape"].dtype, device=data["shape"].device),
            "av_idx": -1,
        }
        return out

    # 2) create new tensors by indexing
    # Use .index_select to make it explicit we're creating new tensors
    def sel(x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, keep_idx)

    valid_agent: Dict[str, Any] = {
        "num_nodes": num_valid,
        "valid_mask": sel(data["valid_mask"]),
        "role": sel(data["role"]),
        "id": sel(data["id"]),
        "type": sel(data["type"]),
        "category": sel(data["category"]),
        "position": sel(data["position"]),
        "heading": sel(data["heading"]),
        "velocity": sel(data["velocity"]),
        "shape": sel(data["shape"]),
    }

    # 3) recompute av_idx after filtering (recommended)
    av_idx_old = int(data.get("av_idx", -1))
    if av_idx_old >= 0:
        # map old index -> new index
        # find where keep_idx == av_idx_old
        hit = (keep_idx == av_idx_old).nonzero(as_tuple=False).squeeze(1)
        valid_agent["av_idx"] = int(hit[0].item()) if hit.numel() > 0 else -1
    else:
        valid_agent["av_idx"] = -1


    return valid_agent


def get_map_features_av2(static_map: ArgoverseStaticMap) -> Dict[str, Any]:
    """Extract map graph features into Waymo-style tensors.

    Changes from your original:
      4) `map_point['height']` now matches your Waymo preprocessing semantics: height = Δz (adjacent diff),
         aligned with the vector from point i -> i+1.

    NOTE:
      - We keep `map_point['position']` as the start point of each vector (centerline[:-1]).
      - `height` is now vectors[:,2] (delta-z), NOT absolute z.
    """

    polygon_ids = []
    polygon_types = []

    all_points = []  # start points of each vector: [N_vec, 3]
    all_point_orientations = []
    all_point_magnitudes = []
    all_point_types = []
    all_point_heights = []  # Δz per vector

    point_to_polygon_edge_index_list = []
    polygon_to_polygon_edge_index_list = []
    polygon_to_polygon_type_list = []

    current_point_idx = 0
    current_poly_idx = 0

    # --- 1. Lanes ---
    lane_ids = sorted(static_map.vector_lane_segments.keys())

    for lane_id in lane_ids:
        lane_seg = static_map.vector_lane_segments[lane_id]

        polygon_ids.append(lane_id)
        w_type = AV2_LANE_TYPE_TO_ID.get(lane_seg.lane_type, _POLYGON_TYPES.index('VEHICLE'))
        polygon_types.append(w_type)

        centerline = static_map.get_lane_segment_centerline(lane_id)  # [N,3]
        num_pts = len(centerline)
        if num_pts < 2:
            # Still advance polygon index because we already appended this polygon
            current_poly_idx += 1
            continue

        start_pts = centerline[:-1]
        vectors = centerline[1:] - centerline[:-1]

        mags = np.linalg.norm(vectors[:, :2], axis=1)
        oris = np.arctan2(vectors[:, 1], vectors[:, 0])
        dz = vectors[:, 2]

        all_points.append(start_pts)
        all_point_magnitudes.append(mags)
        all_point_orientations.append(oris)
        all_point_heights.append(dz)
        all_point_types.append(np.full(num_pts - 1, POINT_TYPE_CENTERLINE, dtype=np.uint8))

        p_indices = np.arange(current_point_idx, current_point_idx + num_pts - 1)
        poly_indices = np.full(num_pts - 1, current_poly_idx)
        point_to_polygon_edge_index_list.append(np.stack([p_indices, poly_indices]))

        current_point_idx += (num_pts - 1)
        current_poly_idx += 1

    # --- 2. Crosswalks ---
    cw_ids = sorted(static_map.vector_pedestrian_crossings.keys())
    for cw_id in cw_ids:
        cw = static_map.vector_pedestrian_crossings[cw_id]

        polygon_ids.append(cw_id)
        polygon_types.append(_POLYGON_TYPES.index('PEDESTRIAN'))

        poly = cw.polygon  # [N,3]
        num_pts = len(poly)
        if num_pts < 2:
            current_poly_idx += 1
            continue

        start_pts = poly[:-1]
        vectors = poly[1:] - poly[:-1]

        mags = np.linalg.norm(vectors[:, :2], axis=1)
        oris = np.arctan2(vectors[:, 1], vectors[:, 0])
        dz = vectors[:, 2]

        all_points.append(start_pts)
        all_point_magnitudes.append(mags)
        all_point_orientations.append(oris)
        all_point_heights.append(dz)
        all_point_types.append(np.full(num_pts - 1, POINT_TYPE_CROSSWALK, dtype=np.uint8))

        p_indices = np.arange(current_point_idx, current_point_idx + num_pts - 1)
        poly_indices = np.full(num_pts - 1, current_poly_idx)
        point_to_polygon_edge_index_list.append(np.stack([p_indices, poly_indices]))

        current_point_idx += (num_pts - 1)
        current_poly_idx += 1

    # --- 3. Connectivity (Lane graph only) ---
    id_to_idx = {pid: i for i, pid in enumerate(polygon_ids)}

    for lane_id in lane_ids:
        curr_idx = id_to_idx.get(lane_id)
        if curr_idx is None:
            continue

        lane_seg = static_map.vector_lane_segments[lane_id]

        for pred_id in lane_seg.predecessors:
            pred_idx = id_to_idx.get(pred_id)
            if pred_idx is not None:
                polygon_to_polygon_edge_index_list.append([pred_idx, curr_idx])
                polygon_to_polygon_type_list.append(_POLYGON_TO_POLYGON_TYPES.index('PRED'))

        for succ_id in lane_seg.successors:
            succ_idx = id_to_idx.get(succ_id)
            if succ_idx is not None:
                polygon_to_polygon_edge_index_list.append([succ_idx, curr_idx])
                polygon_to_polygon_type_list.append(_POLYGON_TO_POLYGON_TYPES.index('SUCC'))

        if lane_seg.left_neighbor_id:
            left_idx = id_to_idx.get(lane_seg.left_neighbor_id)
            if left_idx is not None:
                polygon_to_polygon_edge_index_list.append([left_idx, curr_idx])
                polygon_to_polygon_type_list.append(_POLYGON_TO_POLYGON_TYPES.index('LEFT'))

        if lane_seg.right_neighbor_id:
            right_idx = id_to_idx.get(lane_seg.right_neighbor_id)
            if right_idx is not None:
                polygon_to_polygon_edge_index_list.append([right_idx, curr_idx])
                polygon_to_polygon_type_list.append(_POLYGON_TO_POLYGON_TYPES.index('RIGHT'))

    # --- Assemble tensors ---
    if all_points:
        points_tensor = torch.from_numpy(np.concatenate(all_points, axis=0)).float()
        orientations_tensor = torch.from_numpy(np.concatenate(all_point_orientations, axis=0)).float()
        mags_tensor = torch.from_numpy(np.concatenate(all_point_magnitudes, axis=0)).float()
        types_tensor = torch.from_numpy(np.concatenate(all_point_types, axis=0))
        heights_tensor = torch.from_numpy(np.concatenate(all_point_heights, axis=0)).float()
    else:
        points_tensor = torch.zeros((0, 3), dtype=torch.float)
        orientations_tensor = torch.zeros((0,), dtype=torch.float)
        mags_tensor = torch.zeros((0,), dtype=torch.float)
        types_tensor = torch.zeros((0,), dtype=torch.uint8)
        heights_tensor = torch.zeros((0,), dtype=torch.float)

    if point_to_polygon_edge_index_list:
        p2poly_edge_index = torch.from_numpy(np.concatenate(point_to_polygon_edge_index_list, axis=1)).long()
    else:
        p2poly_edge_index = torch.tensor([[], []], dtype=torch.long)

    if polygon_to_polygon_edge_index_list:
        poly2poly_edge_index = torch.tensor(polygon_to_polygon_edge_index_list).t().long()
        poly2poly_type = torch.tensor(polygon_to_polygon_type_list, dtype=torch.uint8)
    else:
        poly2poly_edge_index = torch.tensor([[], []], dtype=torch.long)
        poly2poly_type = torch.tensor([], dtype=torch.uint8)

    num_polygons = len(polygon_ids)
    return {
        'map_polygon': {
            'num_nodes': num_polygons,
            'type': torch.tensor(polygon_types, dtype=torch.uint8),
            'light_type': torch.ones(num_polygons, dtype=torch.uint8) * 3,
        },
        'map_point': {
            'num_nodes': int(points_tensor.shape[0]),
            'position': points_tensor,
            'orientation': orientations_tensor,
            'magnitude': mags_tensor,
            'height': heights_tensor,  # Δz per vector (Waymo semantics)
            'type': types_tensor,
            'side': torch.zeros(int(points_tensor.shape[0]), dtype=torch.uint8),
        },
        ('map_point', 'to', 'map_polygon'): {
            'edge_index': p2poly_edge_index,
        },
        ('map_polygon', 'to', 'map_polygon'): {
            'edge_index': poly2poly_edge_index,
            'type': poly2poly_type,
        },
    }


def process_single_scenario(log_dir: Path, output_dir: Path):
    """Worker function to process a single scenario directory."""

    scenario_id = log_dir.name
    output_file = output_dir / f"{scenario_id}.pkl"

    if output_file.exists():
        return

    try:
        json_file = log_dir / f"log_map_archive_{scenario_id}.json"
        parquet_file = log_dir / f"scenario_{scenario_id}.parquet"

        static_map = ArgoverseStaticMap.from_json(json_file)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_file)

        # --- Process ---
        # 1) Agents truncated to first 91 steps
        agent_data = get_agent_features_av2(scenario, num_steps=NUM_STEPS)
        # 2) Map with height = Δz (neighbor diff)
        map_data = get_map_features_av2(static_map)

        data: Dict[str, Any] = {}
        data['scenario_id'] = scenario_id
        data['agent'] = agent_data
        data.update(map_data)

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

    except Exception as e:
        print(f"Error processing {scenario_id}: {e}")


def batch_process(data_root: str, output_root: str, split: str, num_workers: int = 16):
    """Parallel batch processing of scenarios."""

    print(f"Initializing Dataset from {data_root}...")
    dataset = AV2BaseDataset(
        data_root=Path(data_root),
        split=split,
    )

    output_dir = Path(output_root) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(dataset)} logs. Processing to {output_dir} with {num_workers} workers...")

    process_func = partial(process_single_scenario, output_dir=output_dir)

    with multiprocessing.Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(process_func, dataset.log_dirs), total=len(dataset)))


def main():
    batch_process(DATA_ROOT, OUTPUT_ROOT, SPLIT, num_workers=64)


if __name__ == "__main__":
    main()
