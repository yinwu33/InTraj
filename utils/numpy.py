import numpy as np
import torch


def to_numpy(arr):
    if torch.is_tensor(arr):
        # numpy doesn't support bfloat16, so cast
        if arr.dtype in (torch.bfloat16, torch.float16):
            arr = arr.float()
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data
