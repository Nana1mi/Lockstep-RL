"""Measure a real full-parameter AdamW step, alone or with idle inference/KV."""

import argparse
import atexit
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from lockstep_rl.logprobs import selected_logprobs


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--co-resident", action="store_true")
    parser.add_argument("--inference-memory-fraction", type=float, default=0.3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    if min(args.batch_size, args.chunk_size, args.steps) < 1 or args.sequence_length < 2:
        parser.error("positive batch/chunk/steps and sequence length >= 2 required")
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**vars(args), "output": str(args.output)}, "status": "running",
              "precision": "FP32 parameters + BF16 autocast; selected scores FP32",
              "optimizer": "AdamW(lr=1e-5, foreach=False)", "steps": []}
    engine = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        record["before"] = memory()
        if args.co_resident:
            from nanovllm import LLM, SamplingParams
            engine = LLM(args.model, enforce_eager=True, max_num_seqs=8,
                         max_num_batched_tokens=512, max_model_len=2048,
                         gpu_memory_utilization=args.inference_memory_fraction)
            result = engine.generate([[42, 43, 44]],
                                     SamplingParams(temperature=1, max_tokens=4, ignore_eos=True),
                                     use_tqdm=False)
            assert len(result[0]["token_ids"]) == 4
            record["inference"] = {
                "kv_bytes": engine.model_runner.kv_cache.numel() * engine.model_runner.kv_cache.element_size(),
                "num_kv_blocks": engine.model_runner.config.num_kvcache_blocks,
                "pre_training_output_ids": result[0]["token_ids"], "memory": memory()}

        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        model.train()
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        torch.cuda.synchronize()
        record["model_load_seconds"] = time.perf_counter() - start
        record["parameter_count"] = sum(p.numel() for p in model.parameters())
        record["loaded"] = memory()
        ids = torch.randint(100, min(10000, model.config.vocab_size),
                            (args.batch_size, args.sequence_length), device="cuda")
        mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
        mask[:, (args.sequence_length - 1) // 2:] = True
        head = model.get_output_embeddings()
        observed = model.model.layers[0].self_attn.q_proj.weight
        for step in range(args.steps):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            before = observed.detach().clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
                scores = selected_logprobs(hidden[:, :-1], head, ids[:, 1:], args.chunk_size)
                loss = -(scores * mask).sum() / mask.sum()
            after_forward = memory()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(loss) or grad_norm <= 0:
                raise RuntimeError("Expected finite loss and nonzero finite gradient")
            after_backward = memory()
            optimizer.step()
            torch.cuda.synchronize()
            delta = (observed.detach() - before).abs().max().item()
            if delta == 0:
                raise RuntimeError("Optimizer step did not change the observed parameter")
            record["steps"].append({"index": step, "loss": loss.item(),
                                    "grad_norm": grad_norm.item(), "max_parameter_delta": delta,
                                    "seconds": time.perf_counter() - start,
                                    "after_forward": after_forward, "after_backward": after_backward,
                                    "after_step": memory(), "valid_tokens": mask.sum().item()})
            save()
            del hidden, scores, loss, before
        if engine is not None:
            result = engine.generate([[42, 43, 44]],
                                     SamplingParams(temperature=1, max_tokens=4, ignore_eos=True),
                                     use_tqdm=False)
            assert len(result[0]["token_ids"]) == 4
            record["inference"]["post_training_output_ids"] = result[0]["token_ids"]
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        if torch.cuda.is_initialized():
            record["failure_memory"] = memory()
        raise
    finally:
        save()
        print(json.dumps(record, indent=2))
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


if __name__ == "__main__":
    main()
