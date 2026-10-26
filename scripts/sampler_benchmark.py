"""Manual GPU microbenchmark of original sampling math vs FP32 token scoring."""

import argparse
import csv
import subprocess
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm.layers.sampler import sample_with_logprobs
from lockstep_rl.diagnostics import distribution
from lockstep_rl.experiment_io import configure_numerics, provenance, save_json


def baseline_sample(logits, temperatures):
    # Exact sampler mathematics from the fixed upstream baseline.
    scaled = logits.float().div_(temperatures.unsqueeze(1))
    probs = torch.softmax(scaled, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)


def measure_batch(args, batch, functions, rows):
    logits = torch.randn(batch, args.vocab_size, device="cuda", dtype=torch.bfloat16)
    temperatures = torch.ones(batch, device="cuda")
    for name, function in functions.items():
        for _ in range(10):
            function(logits, temperatures)
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.iterations):
                function(logits, temperatures)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            rows.append({"batch": batch, "variant": name, "repeat": repeat,
                         "microseconds_per_call": elapsed / args.iterations * 1e6,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    tokens, scores = functions["sampling_and_logprob"](logits, temperatures)
    expected = logits.float().log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    torch.testing.assert_close(scores, expected, atol=2e-5, rtol=1e-5)
    return {"batch": batch, "max_absolute_score_error": float((scores - expected).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sampler-benchmark"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--power-mode", help="Actual user-declared AC/performance mode")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.batches + [args.vocab_size, args.iterations]) < 1 or args.repeats < 3:
        parser.error("positive dimensions/iterations and at least three repeats required")
    if args.dry_run:
        print("GPU_PENDING: synthetic BF16 logits; original sampling vs added FP32 selected-token scoring")
        return
    if not args.allow_gpu:
        parser.error("GPU microbenchmarks require manual --allow-gpu")
    if not args.power_mode:
        parser.error("record the actual --power-mode before performance measurement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory")
    settings = configure_numerics(torch)
    torch.manual_seed(0)
    functions = {"sampling": torch.compile(baseline_sample), "sampling_and_logprob": torch.compile(sample_with_logprobs)}
    rows = []
    report = {"status": "running", "scope": "synthetic-logit sampler compute only; full engine/transport overhead measured separately",
              "provenance": provenance(), "numerical_settings": settings,
              "iterations": args.iterations, "repeats": args.repeats, "power_mode": args.power_mode,
              "metric_label": "Microseconds per sampler call", "accuracy_checks": [], "measurements": rows}
    try:
        for batch in args.batches:
            report["accuracy_checks"].append(measure_batch(args, batch, functions, rows))
        report["groups"] = {
                  f"batch={batch},{variant}": distribution([row["microseconds_per_call"] for row in rows
                     if row["batch"] == batch and row["variant"] == variant])
                  for batch in args.batches for variant in functions}
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(args.output_dir / "manifest.json", report)
        if rows:
            with (args.output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    subprocess.run([sys.executable, "scripts/plot_results.py", "--kind", "benchmark", "--input",
                    str(args.output_dir / "manifest.json"), "--output", str(args.output_dir / "sampler.png")], check=True)


if __name__ == "__main__":
    main()
