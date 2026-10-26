"""Manual GPU benchmark against an explicitly selected unmodified source tree."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lockstep_rl.experiment_io import configure_numerics, provenance, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", choices=["baseline", "current"], required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--backend", choices=["eager", "graph"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--fp32-lm-head", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU benchmarks require manual --allow-gpu")
    if args.fp32_lm_head and args.source != "current":
        parser.error("fp32 head is only available in current source")
    source = args.source_path.resolve()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    case = settings["cases"][args.case]
    sys.path.insert(0, str(source))
    import_start = time.perf_counter()
    import torch
    import nanovllm
    from nanovllm import LLM, SamplingParams
    import_seconds = time.perf_counter() - import_start
    if not Path(nanovllm.__file__).resolve().is_relative_to(source):
        raise RuntimeError("imported a different nanovllm source tree")
    numeric = configure_numerics(torch, compile_enabled=settings.get("torch_compile", True))
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    report = {"source": args.source, "module_path": nanovllm.__file__, "backend": args.backend,
              "fp32_lm_head": args.fp32_lm_head,
              "module_import_seconds": import_seconds,
              "case": args.case, "config": settings, "numerical_settings": numeric,
              "provenance": provenance(settings["model"]), "status": "running"}
    try:
        start = time.perf_counter()
        engine_options = dict(settings["engine"])
        if args.source == "current":
            engine_options["fp32_lm_head"] = args.fp32_lm_head
        engine = LLM(str(Path(settings["model"]).resolve()), enforce_eager=args.backend == "eager", **engine_options)
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - start
        report["cold_scope"] = "new process; disk/compiler caches may be warm; constructor and module imports recorded separately"
        report["kv_blocks"] = engine.model_runner.config.num_kvcache_blocks
        prompts = [[random.randint(100, min(10000, engine.model_runner.config.hf_config.vocab_size - 1))
                    for _ in range(case["input_tokens"])] for _ in range(case["requests"])]
        sampling = SamplingParams(temperature=1, max_tokens=case["output_tokens"], ignore_eos=True)
        start = time.perf_counter()
        engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        report["warmup_seconds"] = time.perf_counter() - start
        report["prompt_token_ids"] = prompts
        report["cache_state"] = "same prompts warmed, prefix cache retained"
        report["measurements"] = []
        for repeat in range(settings["steady_repeats"]):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rows = engine.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            tokens = sum(len(row["token_ids"]) for row in rows)
            if tokens != case["requests"] * case["output_tokens"]:
                raise RuntimeError("output workload did not match the fixed budget")
            report["measurements"].append({"repeat": repeat, "seconds": seconds,
                "output_tokens": tokens, "output_tokens_per_second": tokens / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "response_ids": [row["token_ids"] for row in rows]})
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(args.output, report)
    # Exit through atexit for both code trees, preserving the exact upstream API.


if __name__ == "__main__":
    main()
