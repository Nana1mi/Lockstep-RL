"""CPU-safe experiment provenance and serialization helpers."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance(model_path=None):
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    diff = git("diff", "HEAD")
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version,
              "commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
              "diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest(),
              "diff": diff, "packages": {}}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                             capture_output=True, text=True, check=False, timeout=30)
        result["gpu_inventory"] = {"returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr}
    except FileNotFoundError:
        result["gpu_inventory"] = {"available": False}
    torch_module = sys.modules.get("torch")
    result["torch_cuda_build"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    for name in ("torch", "transformers", "triton", "flash-attn", "numpy"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if model_path:
        if not list(Path(model_path).glob("*.safetensors")):
            raise ValueError("model has no local safetensors weights")
        manifest = Path(model_path) / "_lockstep_model.json"
        if manifest.is_file():
            result["model"] = json.loads(manifest.read_text(encoding="utf-8"))
            if result["model"].get("status") != "complete":
                raise ValueError("model download manifest is incomplete")
        else:
            # Content hashes give existing local checkpoints reproducible identity.
            files = list(Path(model_path).glob("*.safetensors")) + list(Path(model_path).glob("*.json"))
            if not any(file.suffix == ".safetensors" for file in files):
                raise ValueError("model has no local safetensors weights")
            hashes = {}
            for file in sorted(files):
                digest = hashlib.sha256()
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[file.name] = {"bytes": file.stat().st_size, "sha256": digest.hexdigest()}
            result["model"] = {"local_files": hashes, "path": str(Path(model_path).resolve())}
    return result


def configure_numerics(torch, *, compile_enabled=True):
    if not isinstance(compile_enabled, bool):
        raise ValueError("torch_compile must be a boolean")
    stance = "default" if compile_enabled else "force_eager"
    torch.compiler.set_stance(stance)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {"tf32_matmul": False, "tf32_cudnn": False,
            "bf16_reduced_precision_reduction": False,
            "torch_compile_enabled": compile_enabled, "compiler_stance": stance}


def exact_chat_prompt(tokenizer, length, *, filler=" a"):
    """Construct a full thinking-disabled chat template at a chosen token length."""
    count = max(1, length - 32)
    for _ in range(16):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": filler * count}],
                                            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) == length:
            return ids
        count += length - len(ids)
        if count < 1:
            break
    raise ValueError("could not construct the exact chat length; choose a larger length or different filler")
