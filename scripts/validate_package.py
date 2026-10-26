"""Verify wheel contents and isolated CPU imports without installing GPU deps."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = sorted((ROOT / ".build/wheels").glob("lockstep_rl-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise ValueError("build a wheel first")
    wheel = wheels[-1]
    expected = {str(path.relative_to(ROOT)).replace("\\", "/")
                for package in ("nanovllm", "lockstep_rl") for path in (ROOT / package).rglob("*.py")}
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            missing = expected - set(archive.namelist())
            if missing:
                raise ValueError(f"missing package sources: {sorted(missing)}")
            stale = [name for name in expected if archive.read(name) != (ROOT / name).read_bytes()]
            if stale:
                raise ValueError(f"wheel source differs from current worktree: {stale}")
            for name in archive.namelist():
                (Path(directory) / name).resolve().relative_to(Path(directory).resolve())
            archive.extractall(directory)
        code = """
import pathlib, sys, torch
sys.path.insert(0, sys.argv[1])
import nanovllm
from lockstep_rl.grpo import GRPOConfig
from lockstep_rl.checkpoint import save_actor_checkpoint
assert pathlib.Path(nanovllm.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
assert GRPOConfig().group_size == 4
assert not torch.cuda.is_initialized()
"""
        subprocess.run([sys.executable, "-c", code, directory], cwd=directory,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
    result = {"status": "passed", "wheel": wheel.name, "python_sources": len(expected),
              "scope": "all library sources packaged; isolated imports with CPU torch and no FA2/Triton"}
    (ROOT / "results/package-check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
