"""Validate docs, configs and dry-run CLIs without executing GPU/download jobs."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BASH_ENV": "/dev/null"}
    checks = []
    documents = [ROOT / "README.md", ROOT / "GPU_RUNBOOK.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in documents:
        for index, match in enumerate(re.finditer(r"^```(bash|python)\s*\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)):
            language, body = match.groups()
            if language == "python":
                compile(body, str(path), "exec")
            else:
                subprocess.run(["bash", "-n"], input=body, text=True, env=env, check=True)
                for heredoc in re.finditer(r"<<'PY'\s*\n(.*?)\nPY(?:\n|$)", body, re.S):
                    tree = ast.parse(heredoc.group(1))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.BinOp) and isinstance(node.right, ast.Constant) and node.right.value == "\\n":
                            raise ValueError(f"literal backslash-n in documentation writer: {path}")
            checks.append({"document": str(path.relative_to(ROOT)), "block": index, "language": language})
    commands = {
        "diagnostics-dry-run.json": ["scripts/diagnose.py", "--dry-run"],
        "weight-sync-dry-run.json": ["scripts/weight_sync_experiment.py", "--dry-run"],
        "benchmark-dry-run.json": ["scripts/benchmark_suite.py", "--dry-run"],
        "training-dry-run.json": ["scripts/train_grpo.py", "--dry-run"],
        "evaluation-dry-run.json": ["scripts/evaluate.py", "--model", "nonexistent-local-model", "--dry-run"],
        "hf-score-dry-run.json": ["scripts/score_hf.py", "--model", "nonexistent-local-model", "--trajectory", "unused.json", "--output", "unused.json", "--dry-run"],
        "rental-dry-run.json": ["scripts/run_rental.py", "--persistent-root", "/persistent/lockstep", "--dry-run"],
        "demo-lr1-dry-run.json": ["scripts/train_grpo.py", "--config", "configs/train-demo-simple.json", "--dry-run"],
        "demo-lr5-dry-run.json": ["scripts/train_grpo.py", "--config", "configs/train-demo-simple-lr5e-6.json", "--dry-run"],
    }
    for filename, arguments in commands.items():
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        if payload["status"] != "GPU_PENDING":
            raise ValueError(f"dry run did not identify pending GPU work: {filename}")
        (ROOT / "results" / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assets = subprocess.run([sys.executable, "scripts/prepare_rental_assets.py", "--persistent-root", "/persistent/lockstep", "--dry-run"],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    payload = json.loads(assets.stdout)
    if payload["status"] != "DOWNLOAD_PENDING" or payload["gpu_execution"]:
        raise ValueError("asset prefetch dry-run must remain download/GPU-free")
    (ROOT / "results/assets-dry-run.json").write_text(json.dumps(payload, indent=2) + "\n")
    guard = subprocess.run(["bash", "scripts/setup_gpu_env.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
    if guard.returncode != 2 or "--allow-install" not in guard.stderr:
        raise ValueError("GPU installer was not opt-in guarded")
    subprocess.run(["bash", "scripts/setup_gpu_env.sh", "--dry-run"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "scripts/sampler_benchmark.py", "--dry-run"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, env=env, check=True)
    report = {"status": "passed", "documentation_blocks": checks, "dry_runs": commands,
              "gpu_installer_guard": "passed", "dependency_check": "passed",
              "scope": "CPU-only syntax/configuration/entry-point checks; no GPU result"}
    (ROOT / "results/cpu-artifact-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(checks)} documentation blocks and {len(commands)} JSON dry runs")


if __name__ == "__main__":
    main()
