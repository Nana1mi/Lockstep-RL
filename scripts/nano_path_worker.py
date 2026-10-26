"""One isolated manual GPU fixed-trajectory path run; used by diagnose.py."""

import argparse
import atexit
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.diagnostics import path_evidence, require_path
from lockstep_rl.experiment_io import configure_numerics, exact_chat_prompt, provenance, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", choices=["sample", "eager", "prefix", "chunked", "graph", "graph_padding", "batch", "preemption"], required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU path experiments require manual --allow-gpu")
    if args.case != "sample" and not args.trajectory:
        parser.error("replay requires --trajectory")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    report = {"status": "running", "case": args.case, "config": settings}
    engine = None
    try:
        import torch
        from transformers import AutoTokenizer
        from nanovllm import LLM, SamplingParams
        report["provenance"] = provenance(settings["model"])
        report["numerical_settings"] = configure_numerics(torch, compile_enabled=settings.get("torch_compile", True))
        torch.manual_seed(settings["seed"])
        tokenizer = AutoTokenizer.from_pretrained(settings["model"], local_files_only=True)
        if args.case == "sample":
            prompt = exact_chat_prompt(tokenizer, settings["prompt_tokens"])
            forced = None
        else:
            trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
            target = next(row for row in trajectory if row["request_id"] == "target")
            prompt, forced = target["prompt_ids"], target["response_ids"]
        engine_config = {**settings["engine"], "record_trace": True}
        engine_config["enforce_eager"] = args.case not in ("graph", "graph_padding")
        if args.case == "chunked":
            engine_config["max_num_batched_tokens"] = settings["chunk_budget"]
        if args.case == "preemption":
            engine_config["diagnostic_kv_blocks"] = settings["preemption_kv_blocks"]
        report["engine_config"] = engine_config
        engine = LLM(settings["model"], **engine_config)
        metadata = {"enable_thinking": False, "chat_template_version": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()}
        if args.case == "prefix":
            engine.generate([prompt], SamplingParams(ignore_eos=True), request_ids=["warmup"],
                            forced_response_ids=[forced], sampling_config=metadata, use_tqdm=False)
            engine.scheduler.trace.clear()
        prompts, ids = [prompt], ["target"]
        forced_responses = [forced] if forced is not None else None
        extra = 2 if args.case == "graph_padding" else settings["batch_size"] - 1 if args.case == "batch" else 1 if args.case == "preemption" else 0
        for index in range(extra):
            if args.case == "preemption":
                other = exact_chat_prompt(tokenizer, settings["prompt_tokens"], filler=" b")
            else:
                other = tokenizer.apply_chat_template([{"role": "user", "content": "Describe a small tree. " * (index + 1)}],
                                                       tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prompts.append(other); ids.append(f"other-{index}")
            forced_responses.append([42] * len(forced))
        if args.case == "preemption":
            # Target at the tail is the victim when the first request needs a new block.
            prompts.reverse(); ids.reverse(); forced_responses.reverse()
        rows = engine.generate(prompts, SamplingParams(max_tokens=settings["response_tokens"], ignore_eos=True),
                               request_ids=ids, group_ids=ids, forced_response_ids=forced_responses,
                               sampling_config=metadata, use_tqdm=False)
        trace = engine.scheduler.trace
        report["evidence"] = path_evidence(trace)
        if args.case in ("prefix", "chunked", "graph", "graph_padding", "preemption"):
            require_path(trace, args.case)
        if args.case == "preemption" and "target" not in report["evidence"]["re_prefill_requests"]:
            raise RuntimeError("the target request was not actually preempted and re-prefilled")
        target_batches = [len(row["requests"]) for row in trace if row["event"] == "schedule" and not row["is_prefill"]
                          and any(request["request_id"] == "target" for request in row["requests"])]
        report["target_decode_batch_sizes"] = target_batches
        if args.case == "batch" and not any(size > 1 for size in target_batches):
            raise RuntimeError("target did not decode with another request")
        save_json(args.output_dir / "rollouts.json", [r for r in rows if r["request_id"] == "target"])
        save_json(args.output_dir / "all-rollouts.json", rows)
        save_json(args.output_dir / "trace.json", trace)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None:
            save_json(args.output_dir / "trace.json", engine.scheduler.trace)
        save_json(args.output_dir / "report.json", report)
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


if __name__ == "__main__":
    main()
