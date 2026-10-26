"""Ordered persistent-disk GPU handoff. Default is guarded; --dry-run is CPU-only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lockstep_rl.experiment_io import save_json
from lockstep_rl.data import load_tasks, validate_splits


def validate_assets(persistent):
    """Fail before any CUDA stage if paid-run inputs are missing or partial."""
    persistent = Path(persistent)
    ready = json.loads((persistent / "assets-ready.json").read_text())
    model = persistent / "models/Qwen3-0.6B"
    manifest = json.loads((model / "_lockstep_model.json").read_text())
    if ready.get("status") != "downloaded_not_gpu_validated" or manifest.get("status") != "complete":
        raise ValueError("persistent asset/model preparation is incomplete")
    if not any(path.stat().st_size > 8 for path in model.glob("*.safetensors")):
        raise ValueError("no complete local model weight file")
    data = persistent / "data/gsm8k"
    subsets = {name: load_tasks(data / f"{name}.jsonl") for name in ("train", "validation", "diagnostic", "fewshot")}
    for name, minimum in (("train", 64), ("validation", 32), ("diagnostic", 32), ("fewshot", 32)):
        if len(subsets[name]) < minimum:
            raise ValueError(f"persistent {name} subset is too short")
    for left in subsets:
        for right in subsets:
            if left < right:
                validate_splits(subsets[left], subsets[right])
    if not all(row.get("solution", "").strip() for row in subsets["fewshot"]):
        raise ValueError("persistent few-shot examples lack their worked solutions")


def build_plan(persistent, run_id, python, power_mode):
    persistent = Path(persistent).resolve()
    output = persistent / "results" / run_id
    model = persistent / "models/Qwen3-0.6B"
    data = persistent / "data/gsm8k"
    names = ["diagnostics", "weight-sync", "inference-benchmark", "train-smoke-simple", "train-smoke-gsm8k",
             "train-demo-simple", "train-demo-simple-lr5e-6"]
    configs = {}
    for name in names:
        settings = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        settings["model"] = str(model)
        if "power_mode" in settings:
            settings["power_mode"] = power_mode
        if name in ("diagnostics", "weight-sync"):
            settings.update(question_file=str(data / "diagnostic.jsonl"), fewshot_file=str(data / "fewshot.jsonl"))
        configs[name] = settings
    baseline = json.loads(json.dumps(configs["inference-benchmark"]))
    baseline["cases"] = {"short-single": baseline["cases"]["short-single"]}
    configs["baseline-smoke"] = baseline
    two = json.loads(json.dumps(configs["train-smoke-simple"]))
    two["iterations"] = 2
    two["grpo"]["update_steps"] = 2
    configs["train-two-updates"] = two
    jobs = []
    def add(stage, name, script, *arguments):
        jobs.append({"stage": stage, "name": name, "command": [python, "scripts/" + script, "--allow-gpu", *map(str, arguments)]})
    def config(name):
        return output / "configs" / f"{name}.json"
    add("environment", "environment", "environment_probe.py", "--output", output / "environment.json")
    add("environment", "attention", "attention_smoke.py", "--output", output / "attention.json")
    add("baseline", "baseline", "benchmark_suite.py", "--config", config("baseline-smoke"), "--output-dir", output / "baseline")
    add("diagnostics", "diagnostics", "diagnose.py", "--config", config("diagnostics"), "--output-dir", output / "diagnostics")
    add("weight_sync", "weight-sync", "weight_sync_experiment.py", "--config", config("weight-sync"), "--output-dir", output / "weight-sync")
    add("grpo_smoke_and_two_demos", "two-updates", "train_grpo.py", "--config", config("train-two-updates"), "--anchor", "recomputed", "--output-dir", output / "two-updates")
    for anchor in ("rollout", "recomputed"):
        add("grpo_smoke_and_two_demos", "smoke-" + anchor, "train_grpo.py", "--config", config("train-smoke-simple"),
            "--anchor", anchor, "--output-dir", output / ("smoke-" + anchor))
    for name in ("train-demo-simple", "train-demo-simple-lr5e-6"):
        add("grpo_smoke_and_two_demos", name, "train_grpo.py", "--config", config(name), "--output-dir", output / name)
    add("grpo_smoke_and_two_demos", "gsm8k-smoke", "train_grpo.py", "--config", config("train-smoke-gsm8k"),
        "--train-file", data / "train.jsonl", "--validation-file", data / "validation.jsonl", "--output-dir", output / "gsm8k-smoke")
    add("performance", "performance", "benchmark_suite.py", "--config", config("inference-benchmark"), "--output-dir", output / "performance")
    add("performance", "sampler", "sampler_benchmark.py", "--power-mode", power_mode, "--output-dir", output / "sampler")
    rental = json.loads((ROOT / "configs/rental.json").read_text())
    low = sum(value[0] for value in rental["estimated_gpu_hours"].values())
    high = sum(value[1] for value in rental["estimated_gpu_hours"].values())
    return {"status": "GPU_PENDING", "output": str(output), "jobs": jobs, "configs": configs,
            "hardware": rental["gpu_name"], "estimate": {"base_gpu_hours": [low, high],
            "with_30_percent_contingency": [round(low * 1.3, 2), round(high * 1.3, 2)],
            "scope": "Unmeasured planning range; no installation/download time; wall-clock rental time includes CPU overhead"}}


def gpu_inventory(gpu_id):
    result = subprocess.run(["nvidia-smi", "--id=" + gpu_id,
        "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("select exactly one physical rental GPU")
    name, uuid, driver, memory = [item.strip() for item in rows[0].split(",")]
    return {"name": name, "uuid": uuid, "driver": driver, "memory": memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--run-id", default="review-01")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--power-mode", default="USER_MUST_RECORD_AC_AND_PERFORMANCE_MODE")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must be a simple name within the persistent results directory")
    plan = build_plan(args.persistent_root, args.run_id, args.python, args.power_mode)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if not args.allow_gpu:
        parser.error("rental suite requires manual --allow-gpu; --dry-run is CPU-only")
    if args.power_mode.startswith("USER_MUST"):
        parser.error("record actual rental power/performance mode")
    persistent = args.persistent_root.resolve()
    for name in ("assets-ready.json", "models/Qwen3-0.6B/_lockstep_model.json", "data/gsm8k/manifest.json"):
        if not (persistent / name).is_file():
            parser.error(f"prepare all assets on the persistent disk before renting: missing {name}")
    validate_assets(persistent)
    inventory = gpu_inventory(args.gpu_id)
    if inventory["name"] != plan["hardware"]:
        parser.error("all contrasts must use the selected A100-SXM4-80GB model; do not silently mix hardware")
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        parser.error("use a new run-id; previous raw results must remain intact")
    (output / "configs").mkdir(parents=True, exist_ok=True)
    for name, config in plan["configs"].items():
        save_json(output / "configs" / f"{name}.json", config)
    manifest = {**plan, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": inventory, "completed_jobs": []}
    save_json(output / "manifest.json", manifest)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_id}
    start = time.perf_counter()
    try:
        for job in plan["jobs"]:
            if gpu_inventory(args.gpu_id) != inventory:
                raise RuntimeError("GPU identity/driver changed between measurements")
            before = time.perf_counter()
            with (output / (job["name"] + ".console.log")).open("w") as log:
                result = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest["completed_jobs"].append({**job, "returncode": result.returncode,
                                               "wall_seconds": time.perf_counter() - before})
            save_json(output / "manifest.json", manifest)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; dependent stages stopped; raw output retained")
        manifest["status"] = "executed_review_raw_results"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_clock_gpu_hours"] = (time.perf_counter() - start) / 3600
        save_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
