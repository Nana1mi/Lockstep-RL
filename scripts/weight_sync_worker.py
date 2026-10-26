"""One isolated manual A-to-B update or fresh-B recovery measurement."""

import argparse
import atexit
import hashlib
import json
import sys
import time
import traceback
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.diagnostics import path_evidence, require_path
from lockstep_rl.experiment_io import configure_numerics, provenance, save_json


def parameter_hashes(model):
    import torch
    result = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        digest = hashlib.sha256()
        for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--mode", choices=["hot", "fresh"], required=True)
    parser.add_argument("--backend", choices=["eager", "graph"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--source", choices=["cpu-bf16", "gpu-fp32"], default="gpu-fp32")
    parser.add_argument("--skip-invalidate", action="store_true", help="TEST ONLY: deliberate stale-cache negative control")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU weight-sync experiments require manual --allow-gpu")
    if args.skip_invalidate and args.mode != "hot":
        parser.error("--skip-invalidate is a hot-update negative control only")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    rows = json.loads(args.trajectory.read_text(encoding="utf-8"))
    target = rows[0]
    report = {"status": "running", "mode": args.mode, "backend": args.backend, "config": settings,
              "source": args.source, "negative_control": args.skip_invalidate}
    engine = None
    try:
        import torch
        from nanovllm import LLM, SamplingParams
        from lockstep_rl.scoring import load_effective_hf
        report["provenance"] = provenance(settings["model"])
        report["model_b_provenance"] = provenance(args.model_b)["model"]
        report["numerical_settings"] = configure_numerics(torch, compile_enabled=settings.get("torch_compile", True))
        torch.manual_seed(settings["seed"])
        engine_config = {**settings["engine"], "enforce_eager": args.backend == "eager", "record_trace": True}
        source = None
        if args.mode == "hot":
            preparation = time.perf_counter()
            source = load_effective_hf(args.model_b,
                dtype=torch.float32 if args.source == "gpu-fp32" else torch.bfloat16,
                device="cuda" if args.source == "gpu-fp32" else "cpu")
            if args.source == "gpu-fp32":
                torch.cuda.synchronize()
            report["source_preparation_seconds"] = time.perf_counter() - preparation
            engine = LLM(settings["model"], **engine_config)
            for label in ("A-first", "A-hit"):
                engine.scheduler.trace.clear()
                engine.generate([target["prompt_ids"]], SamplingParams(ignore_eos=True), request_ids=[label],
                                forced_response_ids=[target["response_ids"]], use_tqdm=False)
            require_path(engine.scheduler.trace, "prefix")
            save_json(args.output_dir / "A-cache-hit-trace.json", engine.scheduler.trace)
            before_storage = {name: (p.data_ptr(), tuple(p.shape), str(p.dtype), tuple(p.stride()))
                              for name, p in engine.model_runner.model.named_parameters(remove_duplicate=False)}
            kv_storage = engine.model_runner.kv_cache.data_ptr()
            graph_buffers = {name: tensor.data_ptr() for name, tensor in getattr(engine.model_runner, "graph_vars", {}).items()}
            graph_objects = {batch: id(graph) for batch, graph in getattr(engine.model_runner, "graphs", {}).items()}
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if args.mode == "hot":
            # Production update_weights has no bypass. Only this diagnostic
            # worker can deliberately replace invalidation for one update.
            control = patch.object(engine.scheduler.block_manager, "invalidate", lambda: None) if args.skip_invalidate else nullcontext()
            with control:
                report["weight_update"] = engine.update_weights(source.state_dict(), measure_changes=True)
        else:
            engine = LLM(args.model_b, **engine_config)
        torch.cuda.synchronize()
        ready = time.perf_counter()
        engine.scheduler.trace.clear()
        output = engine.generate([target["prompt_ids"]], SamplingParams(ignore_eos=True), request_ids=[target["request_id"]],
                                 forced_response_ids=[target["response_ids"]], use_tqdm=False)
        torch.cuda.synchronize()
        verified = time.perf_counter()
        first_trace = list(engine.scheduler.trace)
        evidence = path_evidence(first_trace)
        save_json(args.output_dir / "rollouts.json", output)
        save_json(args.output_dir / "first-B-trace.json", first_trace)
        if evidence["prefix_reused_blocks"] and not args.skip_invalidate:
            raise RuntimeError("first B rollout reused a cache entry from before this version")
        if args.skip_invalidate and not evidence["prefix_reused_blocks"]:
            raise RuntimeError("negative control did not actually reuse stale KV")
        if args.backend == "graph":
            require_path(first_trace, "graph")
        report.update(recovery_to_ready_seconds=ready - start,
                      warmup_generation_seconds=verified - ready,
                      recovery_including_verified_rollout_seconds=verified - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      policy_version=engine.policy_version,
                      first_B_evidence=evidence)
        if args.mode == "hot":
            after_storage = {name: (p.data_ptr(), tuple(p.shape), str(p.dtype), tuple(p.stride()))
                             for name, p in engine.model_runner.model.named_parameters(remove_duplicate=False)}
            if before_storage != after_storage:
                raise RuntimeError("hot update replaced parameter storage/layout")
            if engine.policy_version != 1:
                raise RuntimeError("hot update did not commit exactly one new policy version")
            report["parameter_storage_preserved"] = True
            if kv_storage != engine.model_runner.kv_cache.data_ptr():
                raise RuntimeError("hot update replaced physical KV storage")
            if graph_buffers != {name: tensor.data_ptr() for name, tensor in getattr(engine.model_runner, "graph_vars", {}).items()}:
                raise RuntimeError("hot update changed captured graph buffers")
            if graph_objects != {batch: id(graph) for batch, graph in getattr(engine.model_runner, "graphs", {}).items()}:
                raise RuntimeError("hot update replaced compatible CUDA graphs")
            report.update(kv_storage_preserved=True, compatible_graphs_preserved=True)
        report["parameter_hashes"] = parameter_hashes(engine.model_runner.model)
        save_json(args.output_dir / "rollouts.json", output)
        save_json(args.output_dir / "first-B-trace.json", first_trace)
        engine.scheduler.trace.clear()
        engine.generate([target["prompt_ids"]], SamplingParams(ignore_eos=True), request_ids=["B-hit"],
                        forced_response_ids=[target["response_ids"]], use_tqdm=False)
        report["new_B_cache_evidence"] = require_path(engine.scheduler.trace, "prefix")
        save_json(args.output_dir / "new-B-cache-trace.json", engine.scheduler.trace)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None and report["status"] == "failed":
            save_json(args.output_dir / "failure-trace.json", engine.scheduler.trace)
        save_json(args.output_dir / "report.json", report)
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


if __name__ == "__main__":
    main()
