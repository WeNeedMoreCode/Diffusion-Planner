from copy import copy, deepcopy
import torch

from diffusion_planner.utils.train_utils import openjson

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        # stats moved to device once and cached: .to(device) per call is a
        # small H2D copy each way, pure overhead in the hot loop
        self._device_cache = {}

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    def _stats_for(self, device):
        cached = self._device_cache.get(device)
        if cached is None:
            cached = (self.mean.to(device), self.std.to(device))
            self._device_cache[device] = cached
        return cached

    def __call__(self, data):
        mean, std = self._stats_for(data.device)
        return (data - mean) / std

    def inverse(self, data):
        mean, std = self._stats_for(data.device)
        return data * std + mean

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict
        # per-device cache of moved stats (see StateNormalizer): the dict holds
        # CPU tensors, .to(device) per key per call is a small H2D copy
        self._device_cache = {}

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k]= {"mean": torch.tensor(v["mean"], dtype=torch.float32), "std": torch.tensor(v["std"], dtype=torch.float32)}
        return cls(ndt)

    def _stats_for(self, device):
        cached = self._device_cache.get(device)
        if cached is None:
            cached = {k: (v["mean"].to(device), v["std"].to(device)) for k, v in self._normalization_dict.items()}
            self._device_cache[device] = cached
        return cached

    def __call__(self, data):
        norm_data = copy(data)
        device = next(iter(data.values())).device
        for k, (mean, std) in self._stats_for(device).items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - mean) / std
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        device = next(iter(data.values())).device
        for k, (mean, std) in self._stats_for(device).items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * std + mean
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self):
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}