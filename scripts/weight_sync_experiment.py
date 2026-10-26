"""Manual A/B hot-vs-fresh verification, prelocked tolerances and timing repeats."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.diagnostics import compare_trajectories, distribution
from lockstep_rl.experiment_io import provenance, save_json


def build_plan(config, output, repeats):
    root = Path(output)
    jobs = [{"name": "sample", "command": [sys.executable, "scripts/nano_path_worker.py", "--allow-gpu",
             "--config", str(config), "--case", "sample", "--output-dir", str(root / "sample")]}]
    for backend in ("eager", "graph"):
        for mode in ("fresh", "hot"):
            for repeat in range(repeats):
                name = f"{backend}-{mode}-{repeat}"
                jobs.append({"name": name, "backend": backend, "mode": mode, "repeat": repeat,
                    "command": [sys.executable, "scripts/weight_sync_worker.py", "--allow-gpu", "--config", str(config),
                    "--model-b", str(root / "model-B"), "--trajectory", str(root / "sample/rollouts.json"),
                    "--mode", mode, "--backend", backend, "--output-dir", str(root / name)]})
    return jobs


def assert_outputs_close(left, right, *, atol, rtol):
    statistics = compare_trajectories(left, right)
    reference_by_id = {row["request_id"]: row for row in right}
    for a in left:
        b = reference_by_id[a["request_id"]]
        for x, y in zip(a["response_logprobs"], b["response_logprobs"]):
            if abs(x - y) > atol + rtol * abs(y):
                raise RuntimeError("fixed-trajectory scores exceed the predeclared tolerance")
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/weight-sync.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/weight-sync"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings["repeats"] < 3 or any(not math.isfinite(settings[key]) or settings[key] < 0
                                     for key in ("absolute_tolerance", "relative_tolerance")):
        parser.error("at least three repeats and nonnegative predeclared tolerances required")
    if not math.isfinite(settings["perturbation"]) or settings["perturbation"] == 0:
        parser.error("B must have a finite nonzero perturbation")
    jobs = build_plan(args.config, args.output_dir, settings["repeats"])
    if args.dry_run:
        print(json.dumps({"status": "GPU_PENDING", "jobs": jobs, "tolerances": {
             "atol": settings["absolute_tolerance"], "rtol": settings["relative_tolerance"]}}, indent=2)); return
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    if settings["power_mode"].startswith("USER_MUST"):
        parser.error("record actual AC/performance mode before measurement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory to retain previous failures/results")
    manifest = {"status": "preparing", "config": settings, "provenance": provenance(settings["model"]), "jobs": []}
    save_json(args.output_dir / "manifest.json", manifest)
    tolerance = {"atol": settings["absolute_tolerance"], "rtol": settings["relative_tolerance"]}
    save_json(args.output_dir / "predeclared-tolerances.json", tolerance)
    try:
        import torch
        from transformers import AutoTokenizer
        from lockstep_rl.scoring import load_effective_hf
        model_b = load_effective_hf(settings["model"], dtype=torch.bfloat16, device="cpu")
        with torch.no_grad():
            parameter = model_b.get_parameter(settings["parameter"])
            before = parameter.detach().clone()
            parameter.add_(settings["perturbation"])
            if torch.equal(before, parameter):
                raise ValueError("B perturbation disappeared after BF16 rounding")
        model_b.save_pretrained(args.output_dir / "model-B")
        AutoTokenizer.from_pretrained(settings["model"], local_files_only=True).save_pretrained(args.output_dir / "model-B")
        save_json(args.output_dir / "model-B/_lockstep_model.json", {
            "status": "complete", "base": manifest["provenance"]["model"],
            "parameter": settings["parameter"], "perturbation": settings["perturbation"], "dtype": "bfloat16"})
        del model_b
        reports = {}
        references = {}
        for job in jobs:
            with (args.output_dir / f"{job['name']}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
            manifest["jobs"].append({**job, "returncode": result.returncode})
            save_json(args.output_dir / "manifest.json", manifest)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; inspect its saved log")
            if job["name"] == "sample":
                continue
            folder = args.output_dir / job["name"]
            report = json.loads((folder / "report.json").read_text())
            output = json.loads((folder / "rollouts.json").read_text())
            backend = job["backend"]
            reports[job["name"]] = report
            if backend not in references:
                references[backend] = (output, report["parameter_hashes"])
            else:
                reference, hashes = references[backend]
                if hashes != report["parameter_hashes"]:
                    raise RuntimeError("hot/fresh B parameter values differ")
                stats = assert_outputs_close(output, reference, **tolerance)
                save_json(folder / "vs-fresh.json", stats)
        timings = []
        for name, report in reports.items():
            timings.append({"name": name, "mode": report["mode"], "backend": report["backend"],
                            **{key: report[key] for key in ("recovery_to_ready_seconds", "warmup_generation_seconds",
                               "recovery_including_verified_rollout_seconds", "peak_allocated_bytes", "peak_reserved_bytes")}})
        with (args.output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timings[0])); writer.writeheader(); writer.writerows(timings)
        manifest["timing_groups"] = {f"{backend}-{mode}": distribution([
            row["recovery_including_verified_rollout_seconds"] for row in timings if row["backend"] == backend and row["mode"] == mode])
            for backend in ("eager", "graph") for mode in ("fresh", "hot")}
        manifest["status"] = "completed"
        manifest["metric_label"] = "Seconds to verified generation"
        save_json(args.output_dir / "manifest.json", manifest)
        subprocess.run([sys.executable, "scripts/plot_results.py", "--kind", "benchmark", "--input",
                        str(args.output_dir / "manifest.json"), "--output", str(args.output_dir / "recovery.png")], check=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
