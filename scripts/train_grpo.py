"""Manual single-GPU synchronous GRPO training with disjoint validation."""

import argparse
import atexit
from dataclasses import asdict
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.data import chat_prompts, load_tasks, validate_splits
from lockstep_rl.grpo import GRPOConfig
from lockstep_rl.rewards import numeric_reward, simple_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train-simple.json"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--anchor", choices=["rollout", "recomputed"])
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/train"))
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.anchor:
        settings["grpo"]["anchor"] = args.anchor
    config = GRPOConfig(**settings["grpo"])
    if args.train_file or args.validation_file:
        if not args.train_file or not args.validation_file:
            parser.error("both local train and validation files are required")
        train, validation = load_tasks(args.train_file), load_tasks(args.validation_file)
    elif settings["task"] == "simple":
        train, validation = simple_tasks(count=settings["train_count"]), simple_tasks("validation", settings["validation_count"])
    else:
        parser.error("GSM8K requires explicit local --train-file and --validation-file")
    validate_splits(train, validation)
    for field in ("iterations", "prompts_per_iteration", "max_tokens", "evaluation_every"):
        if settings[field] < 1:
            parser.error(f"{field} must be positive")
    if settings["prompts_per_iteration"] > len(train):
        parser.error("prompt batch exceeds the training subset")
    expected_accumulation = (settings["prompts_per_iteration"] * config.group_size + config.microbatch_size - 1) // config.microbatch_size
    if config.gradient_accumulation_steps not in (None, expected_accumulation):
        parser.error("gradient accumulation must cover the complete rollout batch")
    if args.dry_run:
        print(json.dumps({"status": "GPU_PENDING", "config": settings, "grpo": asdict(config),
                          "train_count": len(train), "validation_count": len(validation)}, indent=2))
        return
    if not args.allow_gpu:
        parser.error("Real GPU training requires manual --allow-gpu; use --dry-run for CPU validation")
    import torch
    from transformers import AutoModelForCausalLM
    from nanovllm import LLM
    from lockstep_rl.evaluation import evaluate
    from lockstep_rl.trainer import GRPOTrainer, run_iteration
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    engine = LLM(settings["model"], **settings["engine"])
    try:
        model = AutoModelForCausalLM.from_pretrained(settings["model"], dtype=torch.float32,
                                                    attn_implementation="sdpa", local_files_only=True).to("cuda")
        model.config.use_cache = False
        if settings["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], foreach=False)
        trainer = GRPOTrainer(model, optimizer, config=config, pad_token_id=engine.tokenizer.pad_token_id or 0)
        template = engine.tokenizer.chat_template
        metadata = {"enable_thinking": False,
                    "chat_template_version": hashlib.sha256(str(template).encode()).hexdigest(),
                    "training_master_dtype": "float32", "actor_dtype": str(engine.config.hf_config.dtype)}
        initial = evaluate(engine, validation, label="initial", max_tokens=settings["max_tokens"])
        (args.output_dir / "validation-initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
        with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for iteration in range(settings["iterations"]):
                tasks = random.sample(train, settings["prompts_per_iteration"])
                prompts = chat_prompts(engine.tokenizer, tasks)
                def reward_fn(record, index):
                    text = engine.tokenizer.decode(record["response_ids"], skip_special_tokens=True)
                    return numeric_reward(text, tasks[index]["answer"])
                records, metrics = run_iteration(engine, trainer, prompts, reward_fn, iteration=iteration,
                                                 max_tokens=settings["max_tokens"], sampling_config=metadata)
                metrics["iteration"] = iteration
                metrics["mean_reward"] = sum(metrics["rewards"]) / len(records)
                metrics["truncation_fraction"] = sum(r["finish_reason"] == "length" for r in records) / len(records)
                if (iteration + 1) % settings["evaluation_every"] == 0:
                    metrics["validation"] = evaluate(engine, validation, label=str(iteration), max_tokens=settings["max_tokens"])
                (args.output_dir / f"rollouts-{iteration}.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                stream.write(json.dumps(metrics) + "\n"); stream.flush()
                print(json.dumps(metrics))
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == "__main__":
    main()
