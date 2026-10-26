import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise ValueError(f"checkpoint has no safetensors weights: {path}")
    # Keep the original loader's copy path independent of hot-update copying,
    # but preflight a complete TP=1 checkpoint before initializing any weights.
    if all(getattr(module, "tp_size", 1) == 1 for module in model.modules()):
        from nanovllm.utils.weight_update import prepare_weight_update
        state = {}
        for file in files:
            with safe_open(file, "pt", "cpu") as source:
                for key in source.keys():
                    if key in state:
                        raise ValueError(f"duplicate checkpoint weight key: {key}")
                    state[key] = source.get_tensor(key)
        prepare_weight_update(model, state)
        del state
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
