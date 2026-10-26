"""Save committed actor-effective weights without mutating trainer parameters."""

from pathlib import Path

from lockstep_rl.experiment_io import save_json


def save_actor_checkpoint(model, hf_config, tokenizer, destination, *, policy_version):
    from safetensors.torch import save_file
    path = Path(destination)
    if path.exists() and any(path.iterdir()):
        raise ValueError("checkpoint destination must be empty")
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    seen = set()
    for name, tensor in model.state_dict().items():
        storage = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        if storage in seen:
            continue
        seen.add(storage)
        state[name] = tensor.detach().to(device="cpu", dtype=hf_config.dtype).contiguous()
    save_file(state, path / "model.safetensors")
    hf_config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    save_json(path / "_lockstep_model.json", {"status": "complete", "policy_version": policy_version,
                                             "dtype": str(hf_config.dtype), "scope": "actor-effective weights"})
