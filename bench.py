import os
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    if min(args.num_seqs, args.input_len, args.output_len, args.repeats) < 1:
        parser.error("request counts, lengths and repeats must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input + output length must fit max-model-len")
    if not args.fixed_length and min(args.input_len, args.output_len) < 100:
        parser.error("variable-length mode requires maximum lengths >= 100")
    seed(0)
    torch.manual_seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.input_len
    max_ouput_len = args.output_len

    path = os.path.expanduser(args.model)
    start = time.perf_counter()
    llm = LLM(path, enforce_eager=args.eager, max_model_len=args.max_model_len,
              max_num_batched_tokens=args.token_budget, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.memory_fraction)
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - start

    prompt_token_ids = [[randint(0, 10000) for _ in range(max_input_len if args.fixed_length else randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=1, ignore_eos=True, max_tokens=max_ouput_len if args.fixed_length else randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    traces = []
    if args.trace:
        schedule = llm.scheduler.schedule
        def traced_schedule():
            seqs, is_prefill = schedule()
            traces.append({"is_prefill": is_prefill, "batch_size": len(seqs),
                           "graph_batch": None if is_prefill or args.eager or len(seqs) > 512 else
                           next(x for x in llm.model_runner.graph_bs if x >= len(seqs)),
                           "sequences": [{"seq_id": seq.seq_id, "start": seq.num_cached_tokens,
                                          "scheduled_tokens": seq.num_scheduled_tokens,
                                          "context_tokens": len(seq)} for seq in seqs]})
            return seqs, is_prefill
        llm.scheduler.schedule = traced_schedule
    start = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - start
    traces.clear()
    measurements = []
    for repeat in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        total_tokens = sum(len(output["token_ids"]) for output in outputs)
        assert len(outputs) == num_seqs
        assert total_tokens == sum(sp.max_tokens for sp in sampling_params)
        measurements.append({"repeat": repeat, "output_tokens": total_tokens,
                             "seconds": elapsed, "output_tokens_per_second": total_tokens / elapsed,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print(f"Repeat {repeat}: {total_tokens}tok, {elapsed:.3f}s, {total_tokens / elapsed:.2f}tok/s")
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**vars(args), "output": str(args.output)},
              "initialization_seconds": initialization_seconds, "warmup_seconds": warmup_seconds,
              "prompt_token_ids": prompt_token_ids,
              "output_budgets": [sp.max_tokens for sp in sampling_params],
              "cache_state": "same prompts warmed; prefix cache retained between repeats",
              "trace_overhead_included": args.trace, "measurements": measurements, "trace": traces,
              "kv_blocks": llm.model_runner.config.num_kvcache_blocks}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
