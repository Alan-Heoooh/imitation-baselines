import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_stats(
    path,
    func,
    **kwargs
):
    """
    Get the statistics of dataset.

    Args:
      - path: str, the path to the whole body dataset;
      - func: lambda expression, the specific area of interest.
    """
    rec = []
    for scene_folder in tqdm(sorted(os.listdir(path))):
        if scene_folder[:5] != "scene":
            continue
        scene_path = os.path.join(path, scene_folder)
        for record in sorted(os.listdir(scene_path)):
            if os.path.splitext(record)[-1] != '.npy':
                continue
            t = np.load(os.path.join(scene_path, record), allow_pickle=True).item()
            rec.append(torch.from_numpy(func(t)))
    rec = torch.stack(rec)
    mean = rec.mean(dim = 0)
    std = rec.std(dim = 0)
    std = torch.clip(std, 1e-2, 10)
    return {
        'mean': mean,
        'std': std
    }