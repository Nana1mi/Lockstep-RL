"""Manual subprocess-isolated path matrix and three-way HF/nano comparison."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.diagnostics import compare_trajectories
from lockstep_rl.experiment_io import provenance, save_json
from scripts.nano_path_worker import CASES


def build_plan(config_path, output_dir, settings):
    root = Path(output_dir)
    base = root / "sample" / "rollouts.json"
    jobs = []
    for case in ["sample", *settings["cases"]]:
        command = [sys.executable, "scripts/nano_path_worker.py", "--allow-gpu", "--config", str(config_path),
                   "--case", case, "--output-dir", str(root / case)]
        if case != "sample":
            command += ["--trajectory", str(base)]
        jobs.append({"name": case, "command": command,
                     "trajectory_ids": [f"traj-{k}" for k in range(settings["num_trajectories"])],
                     "batching": "single" if case in ("sample", "eager", "repeat", "fp32_head", "fp32_head_cached") else case})
    for dtype in ("bfloat16", "float32"):
        jobs.append({"name": f"hf-{dtype}", "trajectory_ids": [f"traj-{k}" for k in range(settings["num_trajectories"])],
                     "command": [sys.executable, "scripts/score_hf.py", "--allow-gpu",
                     "--model", settings["model"], "--trajectory", str(base), "--dtype", dtype,
                     "--attention-backend", settings["attention_backend"], "--output", str(root / f"hf-{dtype}.json")]})
        if not settings.get("torch_compile", True):
            jobs[-1]["command"].append("--disable-torch-compile")
    return jobs


def validate_settings(settings):
    if settings["engine"].get("kvcache_block_size", 256) != 256:
        raise ValueError("this path matrix requires the baseline 256-token KV block size")
    supported = set(CASES) - {"sample"}
    if set(settings["cases"]) != supported or len(settings["cases"]) != len(supported):
        raise ValueError("diagnostic plan must cover each required path exactly once")
    if settings["num_trajectories"] < max(4, settings["batch_size"]):
        raise ValueError("need enough real trajectories for all multi-request paths")
    if not settings.get("question_file") or not settings.get("fewshot_file"):
        raise ValueError("real local GSM8K questions and solved few-shot files required")
    if settings["prompt_tokens"] % 256 or settings["prompt_tokens"] < 256:
        raise ValueError("prompt must end at a KV block boundary to force decode growth")
    if not 1 < settings["response_tokens"] < 256 or not 0 < settings["chunk_budget"] < settings["prompt_tokens"]:
        raise ValueError("response/chunk settings do not trigger the required paths")
    if settings["batch_size"] < 2 or settings["engine"]["max_num_seqs"] < max(settings["batch_size"], 4):
        raise ValueError("insufficient sequence capacity for batch/graph-padding coverage")
    if settings["engine"]["max_num_batched_tokens"] < 2 * settings["prompt_tokens"]:
        raise ValueError("preemption case must admit both prompts in the same prefill step")
    if settings["engine"]["max_model_len"] < settings["prompt_tokens"] + settings["response_tokens"]:
        raise ValueError("prompt/response budget exceeds the model context configuration")
    if settings["preemption_kv_blocks"] != 2 * (settings["prompt_tokens"] // 256):
        raise ValueError("preemption pool must initially fit exactly two complete prompts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diagnostics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    validate_settings(settings)
    jobs = build_plan(args.config, args.output_dir, settings)
    if args.dry_run:
        coverage = [{"case": job["name"], "request_id": request}
                    for job in jobs for request in job.get("trajectory_ids", [])]
        print(json.dumps({"status": "GPU_PENDING", "jobs": jobs, "coverage": coverage,
                          "num_trajectories": settings["num_trajectories"]}, indent=2)); return
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    if settings["power_mode"].startswith("USER_MUST"):
        parser.error("record actual AC/performance mode in the experiment config first")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory to preserve earlier raw results")
    manifest = {"status": "running", "config": settings, "provenance": provenance(settings["model"]), "jobs": []}
    save_json(args.output_dir / "manifest.json", manifest)
    try:
        for job in jobs:
            log_path = args.output_dir / f"{job['name']}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, check=False)
            manifest["jobs"].append({**job, "returncode": result.returncode, "log": str(log_path)})
            save_json(args.output_dir / "manifest.json", manifest)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; inspect {log_path}")
        def read(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))
        base = read(args.output_dir / "eager" / "rollouts.json")
        comparisons = {f"path-{case}": (base, read(args.output_dir / case / "rollouts.json"))
                       for case in ["sample", *settings["cases"]] if case != "eager"}
        bf16, fp32 = read(args.output_dir / "hf-bfloat16.json"), read(args.output_dir / "hf-float32.json")
        comparisons.update({"nano-vs-hf-bf16": (base, bf16), "nano-vs-hf-fp32": (base, fp32),
                            "nano-fp32-head-vs-hf-fp32": (read(args.output_dir / "fp32_head/rollouts.json"), fp32),
                            "nano-cached-fp32-head-vs-hf-fp32": (read(args.output_dir / "fp32_head_cached/rollouts.json"), fp32),
                            "fp32-head-live-vs-cached": (read(args.output_dir / "fp32_head/rollouts.json"), read(args.output_dir / "fp32_head_cached/rollouts.json")),
                            "hf-bf16-vs-hf-fp32": (bf16, fp32)})
        noise = compare_trajectories(base, read(args.output_dir / "repeat/rollouts.json"), threshold=settings["threshold"])
        save_json(args.output_dir / "noise-floor.json", noise)
        precision = compare_trajectories(bf16, fp32, threshold=settings["threshold"])
        precision_by_id = {row["request_id"]: row["max_absolute_logprob_error"] for row in precision["per_trajectory"]}
        for name, (left, right) in comparisons.items():
            statistics = compare_trajectories(left, right, threshold=settings["threshold"])
            statistics["noise_floor"] = {"comparison": "eager vs repeat",
                "absolute_logprob_error": noise["absolute_logprob_error"],
                "trajectories_above_threshold": noise["trajectories_above_threshold"],
                "per_trajectory": noise["per_trajectory"]}
            statistics["precision_reference"] = {"comparison": "hf-bf16-vs-hf-fp32",
                "absolute_logprob_error": precision["absolute_logprob_error"], "per_trajectory": precision["per_trajectory"],
                "interpretation": "measured BF16/FP32 accuracy scale, not an automatic error verdict/tolerance"}
            statistics["noise_floor"]["discriminative"] = noise["absolute_logprob_error"]["max"] > 0
            noise_by_id = {row["request_id"]: row["max_absolute_logprob_error"] for row in noise["per_trajectory"]}
            for row in statistics["per_trajectory"]:
                row["observed_repeat_max_error"] = noise_by_id[row["request_id"]]
                row["exceeds_observed_repeat_max"] = (row["max_absolute_logprob_error"] > row["observed_repeat_max_error"]
                                                      if row["observed_repeat_max_error"] > 0 else None)
                row["measured_hf_bf16_fp32_max_error"] = precision_by_id[row["request_id"]]
            save_json(args.output_dir / f"{name}.json", statistics)
            with (args.output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(statistics["rows"][0]))
                writer.writeheader(); writer.writerows(statistics["rows"])
            with (args.output_dir / f"{name}-post-eos.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(statistics["rows"][0]))
                writer.writeheader(); writer.writerows(statistics["post_eos"]["rows"])
            subprocess.run([sys.executable, "scripts/plot_results.py", "--kind", "consistency",
                            "--input", str(args.output_dir / f"{name}.json"), "--output", str(args.output_dir / f"{name}.png")], check=True)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
