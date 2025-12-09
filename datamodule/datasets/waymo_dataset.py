import pickle as pkl
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch import Tensor


class WaymoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        is_train: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.is_train = is_train

        self.data_files = list((self.data_root / split).glob("*.pkl"))

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> dict:
        with open(self.data_files[idx], "rb") as f:
            data = pkl.load(f)

        # idx int
        # lg_type int
        # scene_timestep <class 'numpy.int64'> ()
        # num_agents int
        # num_lanes int
        # road_points <class 'numpy.ndarray'> (15, 20, 2)
        # agent_states <class 'numpy.ndarray'> (8, 7)
        # agent_types <class 'numpy.ndarray'> (8, 3)
        # edge_index_lane_to_lane <class 'torch.Tensor'> torch.Size([2, 225])
        # edge_index_agent_to_agent <class 'torch.Tensor'> torch.Size([2, 64])
        # edge_index_lane_to_agent <class 'torch.Tensor'> torch.Size([2, 120])
        # road_connection_types <class 'numpy.ndarray'> (225, 6)

        # convert to torch tensors for downstream code



if __name__ == "__main__":
    ds = WaymoDataset(
        data_root="./data/scenario_dreamer_ae_preprocess_waymo",
        split="val",
        is_train=False,
    )
    print(f"len: {len(ds)}")

    data = ds[0]
