"""TP=1 weight-update preflight and in-place copy using upstream loaders."""

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import nn


@dataclass
class WeightCopy:
    name: str
    parameter: nn.Parameter
    source: torch.Tensor
    loader: object
    shard: object = None


def _storage_key(parameter):
    return (parameter.untyped_storage().data_ptr(), parameter.storage_offset(),
            tuple(parameter.shape), tuple(parameter.stride()), parameter.dtype, parameter.device)


def _finite_after_conversion(source, dtype):
    # Bound temporary conversion/validation allocations for large embeddings.
    rows = source if source.ndim > 0 else source.unsqueeze(0)
    for chunk in rows.split(128, dim=0):
        if not torch.isfinite(chunk.to(dtype=dtype)).all().item():
            return False
    return True


def _equal_after_conversion(left, right, dtype):
    if left.shape != right.shape:
        return False
    if left is right:
        return True
    a = left if left.ndim else left.unsqueeze(0)
    b = right if right.ndim else right.unsqueeze(0)
    return all(torch.equal(x.to(device="cpu", dtype=dtype), y.to(device="cpu", dtype=dtype))
               for x, y in zip(a.split(128), b.split(128)))


def prepare_weight_update(model, state_dict):
    """Validate every key/shard/shape/dtype/alias before touching model storage.

    The caller must keep source parameters unchanged until commit returns. This
    deliberately stores references, not a whole source-weight snapshot.
    Only the first version's TP=1 fixed-layout dense model is supported.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if not parameters:
        raise ValueError("model has no parameters")
    expected = {}
    aliases = defaultdict(list)
    storage_views = {}
    mapping = getattr(model, "packed_modules_mapping", {})
    for name, parameter in parameters.items():
        if (parameter.layout != torch.strided or not parameter.is_floating_point()
                or parameter.device.type == "meta" or parameter.numel() == 0):
            raise ValueError(f"unsupported target shape/layout/dtype/device: {name}")
        key = _storage_key(parameter)
        storage = (parameter.untyped_storage().data_ptr(), parameter.device)
        if storage in storage_views and storage_views[storage] != key:
            raise ValueError("partial/overlapping parameter aliases are unsupported in v1")
        storage_views[storage] = key
        aliases[key].append(name)
        module_path, _, leaf = name.rpartition(".")
        module = model.get_submodule(module_path)
        if getattr(module, "tp_size", 1) != 1:
            raise NotImplementedError("weight updates are validated for TP=1 only")
        packed = [(src, shard) for src, (dst, shard) in mapping.items()
                  if module_path.split(".")[-1] == dst]
        if packed:
            if not hasattr(parameter, "weight_loader"):
                raise ValueError(f"packed parameter has no upstream loader: {name}")
            for source_module, shard in packed:
                if isinstance(shard, str):
                    if shard not in ("q", "k", "v"):
                        raise ValueError(f"unknown packed shard {shard}")
                    size = (module.num_heads if shard == "q" else module.num_kv_heads) * module.head_size
                else:
                    size = module.output_sizes[shard]
                shape = (size,) + tuple(parameter.shape[1:])
                prefix = module_path.rsplit(".", 1)[0] + "." if "." in module_path else ""
                source_name = prefix + source_module + "." + leaf
                expected[source_name] = (name, parameter, shape, shard)
            if sum(item[2][0] for item in expected.values() if item[0] == name) != parameter.shape[0]:
                raise ValueError(f"packed shards do not cover parameter: {name}")
        else:
            expected[name] = (name, parameter, tuple(parameter.shape), None)
    extras = set(state_dict) - set(expected)
    if extras:
        raise ValueError(f"unexpected weight keys: {sorted(extras)}")
    missing = set(expected) - set(state_dict)
    for names in aliases.values():
        supplied = [name for name in names if name in state_dict]
        if supplied:
            missing.difference_update(names)
    if missing:
        raise ValueError(f"missing weight keys: {sorted(missing)}")
    copies = []
    for source_name, source in state_dict.items():
        name, target, shape, shard = expected[source_name]
        if not isinstance(source, torch.Tensor) or tuple(source.shape) != shape:
            raise ValueError(f"incorrect shape/tensor for {source_name}; expected {shape}")
        if source.layout != torch.strided or not source.is_floating_point() or source.device.type == "meta":
            raise ValueError(f"unsupported source dtype/layout/device: {source_name}")
        if not _finite_after_conversion(source.detach(), target.dtype):
            raise ValueError(f"nonfinite or overflowing weight conversion: {source_name}")
        loader = getattr(target, "weight_loader", None)
        copies.append(WeightCopy(name, target, source.detach(), loader, shard))
    for names in aliases.values():
        supplied = [name for name in names if name in state_dict]
        if len(supplied) > 1:
            first = supplied[0]
            for other in supplied[1:]:
                if not _equal_after_conversion(state_dict[first], state_dict[other], parameters[first].dtype):
                    raise ValueError(f"conflicting tied weights: {first}, {other}")
    return copies


@torch.no_grad()
def commit_weight_update(copies):
    """Write only a successfully preflighted plan; caller owns fail-closed state."""
    for item in copies:
        if item.loader is None:
            item.parameter.copy_(item.source)
        elif item.shard is None:
            item.loader(item.parameter, item.source)
        else:
            item.loader(item.parameter, item.source, item.shard)
