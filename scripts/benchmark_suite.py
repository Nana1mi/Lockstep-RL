"""Manual repeated original-source vs full rollout-contract inference benchmark."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lockstep_rl.diagnostics import distribution
from lockstep_rl.experiment_io import provenance, save_json

BASELINE = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"


def materialize_baseline(destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("baseline destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "archive", "--format=tar", BASELINE], cwd=PROJECT, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            for member in archive.getmembers():
                (destination / member.name).resolve().relative_to(destination.resolve())
                if not (member.isfile() or member.isdir()):
                    raise ValueError("baseline archive contains a non-file/non-directory member")
            archive.extractall(destination, filter="data")
    (destination / "BASELINE_COMMIT").write_text(BASELINE + "\n", encoding="utf-8")


def build_plan(config, output, settings):
    root = Path(output)
    jobs = []
    for source, fp32_head in (("baseline", False), ("current", False), ("current", True)):
        path = root / "upstream-source" if source == "baseline" else PROJECT
        for backend in ("eager", "graph"):
            for case in settings["cases"]:
                for repeat in range(settings["process_repeats"]):
                    variant = "current-fp32-head" if fp32_head else source
                    name = f"{variant}-{backend}-{case}-{repeat}"
                    jobs.append({"name": name, "source": source, "backend": backend, "case": case,
                        "source_variant": variant, "fp32_lm_head": fp32_head,
                        "process_repeat": repeat, "command": [sys.executable, "scripts/inference_benchmark_worker.py", "--allow-gpu",
                        "--config", str(config), "--case", case, "--source", source, "--source-path", str(path),
                        "--backend", backend, "--output", str(root / f"{name}.json")]})
                    if fp32_head:
                        jobs[-1]["command"].append("--fp32-lm-head")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inference-benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference-benchmark"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if min(settings["process_repeats"], settings["steady_repeats"]) < 3:
        parser.error("at least three process and steady repeats required")
    for case in settings["cases"].values():
        if min(case.values()) < 1 or case["input_tokens"] + case["output_tokens"] > settings["engine"]["max_model_len"]:
            parser.error("invalid workload shape or context budget")
    jobs = build_plan(args.config, args.output_dir, settings)
    if args.dry_run:
        print(json.dumps({"status": "GPU_PENDING", "baseline": BASELINE, "jobs": jobs}, indent=2)); return
    if not args.allow_gpu:
        parser.error("GPU benchmarks require manual --allow-gpu")
    if settings["power_mode"].startswith("USER_MUST"):
        parser.error("record actual AC/performance mode before measuring")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory")
    manifest = {"status": "running", "baseline": BASELINE, "config": settings,
                "provenance": provenance(settings["model"]), "jobs": []}
    save_json(args.output_dir / "manifest.json", manifest)
    try:
        materialize_baseline(args.output_dir / "upstream-source")
        flat = []
        for job in jobs:
            with (args.output_dir / f"{job['name']}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
            manifest["jobs"].append({**job, "returncode": result.returncode})
            save_json(args.output_dir / "manifest.json", manifest)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; inspect its log")
            record = json.loads((args.output_dir / f"{job['name']}.json").read_text())
            for measurement in record["measurements"]:
                flat.append({"source": job["source"], "source_variant": job["source_variant"],
                    "fp32_lm_head": job["fp32_lm_head"], "backend": job["backend"], "case": job["case"],
                    "process_repeat": job["process_repeat"], "steady_repeat": measurement["repeat"],
                    "initialization_seconds": record["initialization_seconds"], "warmup_seconds": record["warmup_seconds"],
                    **{key: measurement[key] for key in ("seconds", "output_tokens_per_second", "peak_allocated_bytes", "peak_reserved_bytes")}})
        with (args.output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        manifest["groups"] = {f"{variant}-{backend}-{case}": distribution([
            row["output_tokens_per_second"] for row in flat if row["source_variant"] == variant and row["backend"] == backend and row["case"] == case])
            for variant in ("baseline", "current", "current-fp32-head") for backend in ("eager", "graph") for case in settings["cases"]}
        manifest["fp32_head_throughput_change_percent"] = {
            f"{backend}-{case}": 100 * (manifest["groups"][f"current-fp32-head-{backend}-{case}"]["mean"] /
                manifest["groups"][f"current-{backend}-{case}"]["mean"] - 1)
            for backend in ("eager", "graph") for case in settings["cases"]}
        manifest["scope"] = "full rollout-data-contract overhead vs original; steady repeats nested within independent process repeats"
        manifest["metric_label"] = "Output tokens per second"
        manifest["status"] = "completed"
        save_json(args.output_dir / "manifest.json", manifest)
        subprocess.run([sys.executable, "scripts/plot_results.py", "--kind", "benchmark", "--input",
                        str(args.output_dir / "manifest.json"), "--output", str(args.output_dir / "throughput.png")], check=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
