"""Isolated manual multi-trajectory replay; every companion is a real sample."""

import argparse
import atexit
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.diagnostics import path_evidence, require_path
from lockstep_rl.experiment_io import configure_numerics, provenance, save_json

CASES = ("sample", "eager", "repeat", "prefix", "chunked", "graph", "graph_padding", "batch",
         "preemption_cached", "preemption_recompute", "fp32_head")


def replay_groups(rows, case, batch_size):
    """Group real trajectories only, with pairs/triples covering short tails."""
    if not rows:
        raise ValueError("replay requires real nonempty trajectories")
    if case in ("batch", "graph_padding", "preemption_cached", "preemption_recompute") and len(rows) < 4:
        raise ValueError("multi-request diagnostics need at least four distinct trajectories")
    if case.startswith("preemption_"):
        return [[rows[(index + 1) % len(rows)], row] for index, row in enumerate(rows)]
    size = 3 if case == "graph_padding" else batch_size if case == "batch" else 1
    groups = [rows[start:start + size] for start in range(0, len(rows), size)]
    # A singleton tail still participates in a real multi-request replay. Only
    # its first canonical occurrence is included in distribution statistics.
    if size > 1 and len(groups[-1]) == 1:
        groups[-1] += rows[:size - 1]
    if case == "graph_padding" and len(groups[-1]) == 2:
        groups[-1] += rows[:1]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
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
            from lockstep_rl.diagnostic_prompts import gsm8k_prompts
            rows = gsm8k_prompts(tokenizer, question_file=settings["question_file"],
                                 fewshot_file=settings["fewshot_file"], count=settings.get("num_trajectories", 1),
                                 prompt_tokens=settings["prompt_tokens"])
            save_json(args.output_dir / "prompt-provenance.json", rows)
        else:
            rows = json.loads(args.trajectory.read_text(encoding="utf-8"))
        if len(rows) != settings.get("num_trajectories", 1):
            raise ValueError("trajectory count does not match config")
        if {row["request_id"] for row in rows} != {f"traj-{k}" for k in range(len(rows))}:
            raise ValueError("trajectory IDs must be the unique canonical traj-k set")
        engine_config = {**settings["engine"], "record_trace": True,
                         "enable_prefix_caching": args.case in ("prefix", "preemption_cached"),
                         "fp32_lm_head": args.case == "fp32_head",
                         "enforce_eager": args.case not in ("graph", "graph_padding")}
        if args.case == "chunked":
            engine_config["max_num_batched_tokens"] = settings["chunk_budget"]
        if args.case.startswith("preemption_"):
            engine_config["diagnostic_kv_blocks"] = settings["preemption_kv_blocks"]
        report["engine_config"] = engine_config
        report["noise_reference"] = "eager vs repeat (same configuration, separate process)"
        engine = LLM(settings["model"], **engine_config)
        metadata = {"enable_thinking": False, "chat_template_version": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()}
        collected, all_rows, all_trace, seen = [], [], [], set()
        groups = replay_groups(rows, args.case, settings.get("batch_size", 4))
        for index, group in enumerate(groups):
            engine.scheduler.block_manager.invalidate()
            engine.scheduler.trace.clear()
            prompts = [row["prompt_ids"] for row in group]
            ids = [f"{row['request_id']}/run-{index}" for row in group]
            forced = None if args.case == "sample" else [row["response_ids"] for row in group]
            if args.case == "prefix":
                engine.generate(prompts, SamplingParams(ignore_eos=True), request_ids=[x + "/warm" for x in ids],
                                forced_response_ids=forced, sampling_config=metadata, use_tqdm=False)
                engine.scheduler.trace.clear()
            output = engine.generate(prompts, SamplingParams(max_tokens=settings["response_tokens"], ignore_eos=True),
                                     request_ids=ids, group_ids=[r["request_id"] for r in group],
                                     forced_response_ids=forced, sampling_config=metadata, use_tqdm=False)
            mapping = dict(zip(ids, [row["request_id"] for row in group]))
            trace = engine.scheduler.trace
            for event in trace:
                if "request_id" in event:
                    event["request_id"] = mapping.get(event["request_id"], event["request_id"])
                for request in event.get("requests", []):
                    request["request_id"] = mapping.get(request["request_id"], request["request_id"])
                event["replay_group"] = index
            # Persist the failing group before any path assertion can raise.
            save_json(args.output_dir / "groups" / f"{index}-trace.json", trace)
            save_json(args.output_dir / "groups" / f"{index}-rollouts.json", output)
            if args.case in ("prefix", "chunked", "graph", "graph_padding", "preemption_cached", "preemption_recompute"):
                require_path(trace, args.case)
            if args.case.startswith("preemption_"):
                victim = group[-1]["request_id"]
                if victim not in path_evidence(trace)["re_prefill_requests"]:
                    raise RuntimeError(f"real trajectory {victim} was not preempted and re-prefilled")
            if args.case == "batch" and not any(not e["is_prefill"] and len(e["requests"]) > 1
                                                  for e in trace if e["event"] == "schedule"):
                raise RuntimeError("real trajectories did not decode together")
            for record in output:
                record["request_id"] = mapping[record["request_id"]]
                record["replay_group"] = index
                all_rows.append(record)
                if args.case.startswith("preemption_") and record["request_id"] != group[-1]["request_id"]:
                    continue
                if record["request_id"] not in seen:
                    seen.add(record["request_id"])
                    collected.append(record)
            all_trace.extend(trace)
            save_json(args.output_dir / "rollouts.json", collected)
            save_json(args.output_dir / "trace.json", all_trace)
        report["evidence"] = path_evidence(all_trace)
        report["trajectory_count"] = len(collected)
        save_json(args.output_dir / "all-rollouts.json", all_rows)
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
