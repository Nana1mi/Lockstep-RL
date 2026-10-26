"""Collect stage-0 evidence without importing the inference engine."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true", help="Manually enable real GPU checks")
    args = parser.parse_args()
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version,
              "executable": sys.executable,
              "git": command("git", "rev-parse", "HEAD"), "packages": {}}
    for name in ("torch", "triton", "transformers", "flash-attn", "flash-attn-4",
                 "safetensors", "xxhash", "numpy"):
        try:
            record["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][name] = None
    record["nvidia_smi"] = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv")
    try:
        import torch
        record["torch_cuda"] = {"version": torch.version.cuda, "status": "GPU_PENDING"}
        if args.allow_gpu and torch.cuda.is_available():
            record["torch_cuda"].update(available=True, arch_list=torch.cuda.get_arch_list())
            record["torch_cuda"].update(name=torch.cuda.get_device_name(),
                                        capability=torch.cuda.get_device_capability())
            x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            y = x @ x.T
            torch.cuda.synchronize()
            record["torch_cuda"]["bf16_matmul_finite"] = bool(y.isfinite().all())
    except Exception as exc:
        record["torch_error"] = f"{type(exc).__name__}: {exc}"
    for module in ("triton", "transformers", "flash_attn"):
        try:
            imported = importlib.import_module(module)
            record[module] = {"file": imported.__file__}
            if module == "flash_attn":
                record[module]["upstream_entrypoints"] = {
                    name: hasattr(imported, name)
                    for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache")}
        except Exception as exc:
            record[module] = {"error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
