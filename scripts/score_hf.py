"""Manual HF BF16/FP32 scoring of a saved fixed trajectory (no generation)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention-backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if args.dry_run:
        print(json.dumps({"status": "GPU_PENDING", "dtype": args.dtype, "trajectory": str(args.trajectory)}))
        return
    if not args.allow_gpu:
        parser.error("GPU scoring requires manual --allow-gpu")
    import torch
    from lockstep_rl.scoring import load_effective_hf, score_records
    records = json.loads(args.trajectory.read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_effective_hf(args.model, dtype=getattr(torch, args.dtype), device="cuda",
                              attention_backend=args.attention_backend)
    scores = score_records(model, records, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
