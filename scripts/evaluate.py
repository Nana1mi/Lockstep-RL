"""Manual independent validation of a local actor checkpoint; never downloads."""

import argparse
import atexit
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lockstep_rl.data import load_tasks, validate_splits
from lockstep_rl.experiment_io import configure_numerics, provenance, save_json
from lockstep_rl.rewards import simple_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train-smoke-simple.json"))
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--train-file", type=Path, help="Reference split for leakage checks")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.validation_file:
        if not args.train_file:
            parser.error("local validation requires --train-file for a disjointness check")
        tasks = load_tasks(args.validation_file)
        training = load_tasks(args.train_file)
    elif settings["task"] == "simple":
        tasks = simple_tasks("validation", settings["validation_count"], settings.get("simple_operation", "add"))
        training = simple_tasks("train", settings["train_count"], settings.get("simple_operation", "add"))
    else:
        parser.error("GSM8K requires a local --validation-file")
    validate_splits(training, tasks)
    if args.dry_run:
        print(json.dumps({"status": "GPU_PENDING", "model": args.model, "validation_count": len(tasks)})); return
    if not args.allow_gpu:
        parser.error("GPU evaluation requires manual --allow-gpu")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory")
    from nanovllm import LLM
    import torch
    from lockstep_rl.evaluation import evaluate
    torch.manual_seed(settings["seed"])
    numerical_settings = configure_numerics(torch, compile_enabled=settings.get("torch_compile", True))
    save_json(args.output_dir / "provenance.json", provenance(args.model))
    save_json(args.output_dir / "config.json", settings)
    save_json(args.output_dir / "tasks.json", tasks)
    engine = LLM(args.model, **settings["engine"])
    try:
        result = evaluate(engine, tasks, label="standalone", max_tokens=settings["max_tokens"],
                          reward_parser=settings.get("reward_parser", "strict"))
        result["scope"] = "single-seed stochastic T=1 validation; no learning-quality guarantee"
        result.update(seed=settings["seed"], numerical_settings=numerical_settings)
        save_json(args.output_dir / "validation.json", result)
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == "__main__":
    main()
