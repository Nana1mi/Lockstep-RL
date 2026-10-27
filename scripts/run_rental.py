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
from lockstep_rl.process_runner import run_task, task_budget
from lockstep_rl.data import load_tasks, validate_splits

STAGES = ("environment", "baseline", "diagnostics", "weight_sync", "grpo", "performance")
STAGE_ALIASES = {"sync": "weight_sync", "grpo_smoke_and_two_demos": "grpo"}


def validate_assets(persistent, splits=None):
    """Fail before any CUDA stage if paid-run inputs are missing or partial."""
    persistent = Path(persistent)
    model = persistent / "models/Qwen3-0.6B"
    manifest = json.loads((model / "_lockstep_model.json").read_text())
    # The global ready marker is informational: a dataset download failure must
    # not prevent model-only baseline/performance jobs from running.
    if manifest.get("status") != "complete":
        raise ValueError("persistent asset/model preparation is incomplete")
    if not any(path.stat().st_size > 8 for path in model.glob("*.safetensors")):
        raise ValueError("no complete local model weight file")
    data = persistent / "data/gsm8k"
    selected = ("train", "validation", "diagnostic", "fewshot") if splits is None else splits
    subsets = {name: load_tasks(data / f"{name}.jsonl") for name in selected}
    for name, minimum in (("train", 64), ("validation", 32), ("diagnostic", 32), ("fewshot", 32)):
        if name in subsets and len(subsets[name]) < minimum:
            raise ValueError(f"persistent {name} subset is too short")
    for left in subsets:
        for right in subsets:
            if left < right:
                validate_splits(subsets[left], subsets[right])
    if not all(row.get("solution", "").strip() for row in subsets.get("fewshot", [])):
        raise ValueError("persistent few-shot examples lack their worked solutions")


def build_plan(persistent, run_id, python, power_mode, stages=None):
    requested = {STAGE_ALIASES.get(stage, stage) for stage in (STAGES if stages is None else stages)}
    if not requested or requested - set(STAGES):
        raise ValueError("select known nonempty stages")
    selected = [stage for stage in STAGES if stage in requested or stage == "environment"]
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
        stage = STAGE_ALIASES.get(stage, stage)
        if stage in selected:
            splits = ["diagnostic", "fewshot"] if stage in ("diagnostics", "weight_sync") else ["train", "validation"] if name == "gsm8k-smoke" else []
            jobs.append({"stage": stage, "name": name, "hard_gate": name in ("environment", "attention"),
                         "asset_splits": splits, **task_budget(name),
                         "command": [python, "scripts/" + script, "--allow-gpu", *map(str, arguments)]})
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
    estimates = {STAGE_ALIASES.get(stage, stage): value for stage, value in rental["estimated_gpu_hours"].items()}
    low = sum(estimates[stage][0] for stage in selected)
    high = sum(estimates[stage][1] for stage in selected)
    return {"status": "GPU_PENDING", "output": str(output), "jobs": jobs, "configs": configs,
            "selected_stages": selected, "failure_policy": "only environment/attention hard-gate; other job failures/timeouts continue after process-tree cleanup; cleanup/inventory failures stop as infrastructure failures",
            "hardware": rental["gpu_name"], "estimate": {"base_gpu_hours": [low, high],
            "with_30_percent_contingency": [round(low * 1.3, 2), round(high * 1.3, 2)],
            "scope": "Unmeasured planning range; no installation/download time; wall-clock rental time includes CPU overhead"}}


def gpu_inventory(gpu_id):
    result = subprocess.run(["nvidia-smi", "--id=" + gpu_id,
        "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("select exactly one physical rental GPU")
    name, uuid, driver, memory = [item.strip() for item in rows[0].split(",")]
    return {"name": name, "uuid": uuid, "driver": driver, "memory": memory}


def execute_jobs(plan, inventory, env, gpu_id, *, validate_inputs=True, inventory_reader=None):
    """Separate failure policy from CUDA work, so the full control flow is CPU-testable."""
    output = Path(plan["output"])
    manifest = {**plan, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": inventory, "completed_jobs": [], "failed_jobs": [], "skipped_jobs": []}
    reader = inventory_reader or gpu_inventory
    start = time.perf_counter()
    try:
        for index, job in enumerate(plan["jobs"]):
            if reader(gpu_id) != inventory:
                raise RuntimeError("GPU identity/driver changed; measurements cannot be mixed")
            before = time.perf_counter()
            error = None
            task_record = {}
            log_path = output / (job["name"] + ".console.log")
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    if validate_inputs and not job["hard_gate"]:
                        validate_assets(output.parent.parent, job["asset_splits"])
                    result = run_task(job["command"], timeout_seconds=job["timeout_seconds"],
                                      cwd=ROOT, env=env, stdout=log, nccl=True)
                    task_record = result.record()
                    error = result.error
                    code = result.returncode
                except Exception as exc:
                    code = 1
                    error = f"{type(exc).__name__}: {exc}"
                    log.write(error + "\n")
            manifest["completed_jobs"].append({**job, **task_record, "returncode": code, "error": error,
                                               "wall_seconds": time.perf_counter() - before})
            if code:
                manifest["failed_jobs"].append(job["name"])
                if job["hard_gate"]:
                    manifest["status"] = "blocked_by_environment"
                    manifest["skipped_jobs"] = [{"name": later["name"], "reason": f"hard gate {job['name']} failed"}
                                                 for later in plan["jobs"][index + 1:]]
                    break
            save_json(output / "manifest.json", manifest)
        else:
            manifest["status"] = "completed_with_failures" if manifest["failed_jobs"] else "executed_review_raw_results"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_failure",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_clock_gpu_hours"] = (time.perf_counter() - start) / 3600
        save_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--run-id", default="review-01")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--power-mode", default="USER_MUST_RECORD_AC_AND_PERFORMANCE_MODE")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=[*STAGES, *STAGE_ALIASES],
                        help="Selected independent stages; environment/attention are always included")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must be a simple name within the persistent results directory")
    plan = build_plan(args.persistent_root, args.run_id, args.python, args.power_mode, args.stages)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if not args.allow_gpu:
        parser.error("rental suite requires manual --allow-gpu; --dry-run is CPU-only")
    if args.power_mode.startswith("USER_MUST"):
        parser.error("record actual rental power/performance mode")
    inventory = gpu_inventory(args.gpu_id)
    if inventory["name"] != plan["hardware"]:
        parser.error("all contrasts must use the selected A100-SXM4-80GB model; do not silently mix hardware")
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        parser.error("use a new run-id; previous raw results must remain intact")
    (output / "configs").mkdir(parents=True, exist_ok=True)
    for name, config in plan["configs"].items():
        save_json(output / "configs" / f"{name}.json", config)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_id}
    manifest = execute_jobs(plan, inventory, env, args.gpu_id)
    print(json.dumps({"status": manifest["status"], "failed_jobs": manifest["failed_jobs"], "manifest": str(output / "manifest.json")}))
    if manifest["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
