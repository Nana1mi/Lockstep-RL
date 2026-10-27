"""Manual single-GPU synchronous GRPO training with disjoint validation."""

import argparse
import atexit
import csv
from dataclasses import asdict
import hashlib
import json
import math
import random
import traceback
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_rl.process_runner import run_checked
from lockstep_rl.data import chat_prompts, load_tasks, validate_splits
from lockstep_rl.grpo import GRPOConfig
from lockstep_rl.rewards import reward_details, simple_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train-smoke-simple.json"))
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--anchor", choices=["rollout", "recomputed"])
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/train"))
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings.get("reward_parser", "strict") not in ("strict", "lenient"):
        parser.error("reward_parser must be strict or lenient")
    if args.anchor:
        settings["grpo"]["anchor"] = args.anchor
    config = GRPOConfig(**settings["grpo"])
    if args.train_file or args.validation_file:
        if not args.train_file or not args.validation_file:
            parser.error("both local train and validation files are required")
        train, validation = load_tasks(args.train_file), load_tasks(args.validation_file)
        settings["train_file"] = str(args.train_file)
        settings["validation_file"] = str(args.validation_file)
        manifest_path = args.train_file.parent / "manifest.json"
        if manifest_path.is_file():
            settings["dataset_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif settings["task"] == "simple":
        operation = settings.get("simple_operation", "add")
        train, validation = simple_tasks(count=settings["train_count"], operation=operation), simple_tasks("validation", settings["validation_count"], operation)
    else:
        parser.error("GSM8K requires explicit local --train-file and --validation-file")
    validate_splits(train, validation)
    for field in ("iterations", "prompts_per_iteration", "max_tokens", "evaluation_every"):
        if settings[field] < 1:
            parser.error(f"{field} must be positive")
    if not math.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        parser.error("learning_rate must be finite and positive")
    if settings.get("optimizer", "adamw") not in ("adamw", "sgd"):
        parser.error("optimizer must be adamw or sgd")
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory to preserve earlier metrics/rollouts")
    import torch
    from transformers import AutoModelForCausalLM
    from nanovllm import LLM
    from lockstep_rl.evaluation import evaluate
    from lockstep_rl.trainer import GRPOTrainer, run_iteration
    from lockstep_rl.experiment_io import configure_numerics, provenance, save_json
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    numerical_settings = configure_numerics(torch, compile_enabled=settings.get("torch_compile", True))
    numerical_settings.update(trainer_autocast_bf16=config.autocast_bf16, actor_autocast=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "provenance.json", provenance(settings["model"]))
    save_json(args.output_dir / "numerical-settings.json", numerical_settings)
    save_json(args.output_dir / "train-tasks.json", train)
    save_json(args.output_dir / "validation-tasks.json", validation)
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    engine = LLM(settings["model"], **settings["engine"])
    try:
        model = AutoModelForCausalLM.from_pretrained(settings["model"], dtype=torch.float32,
                                                    attn_implementation="sdpa", local_files_only=True).to("cuda")
        model.config.use_cache = False
        if settings["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer_class = torch.optim.AdamW if settings.get("optimizer", "adamw") == "adamw" else torch.optim.SGD
        optimizer = optimizer_class(model.parameters(), lr=settings["learning_rate"], foreach=False)
        trainer = GRPOTrainer(model, optimizer, config=config, pad_token_id=engine.tokenizer.pad_token_id or 0)
        template = engine.tokenizer.chat_template
        metadata = {"enable_thinking": False,
                    "chat_template_version": hashlib.sha256(str(template).encode()).hexdigest(),
                    "training_master_dtype": "float32", "actor_dtype": str(engine.config.hf_config.dtype)}
        from nanovllm import SamplingParams
        sync_probe = engine.generate(chat_prompts(engine.tokenizer, train[:1]),
            SamplingParams(max_tokens=settings.get("sync_probe_tokens", 16), ignore_eos=True),
            request_ids=["sync-probe:initial"], use_tqdm=False)[0]
        save_json(args.output_dir / "sync-probe-trajectory.json", sync_probe)
        initial = evaluate(engine, validation, label="initial", max_tokens=settings["max_tokens"],
                           reward_parser=settings.get("reward_parser", "strict"))
        (args.output_dir / "validation-initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
        summary = {"optimizer_steps": 0, "policy_commits": 0, "skipped_batches": 0,
                   "actor_visible_commits": 0, "probe_visible_commits": 0,
                   "iterations": settings["iterations"], "seed": settings["seed"], "scope": "single seed"}
        with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream, (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as csv_stream:
            writer = None
            for iteration in range(settings["iterations"]):
                tasks = random.sample(train, settings["prompts_per_iteration"])
                prompts = chat_prompts(engine.tokenizer, tasks)
                def reward_fn(record, index):
                    text = engine.tokenizer.decode(record["response_ids"], skip_special_tokens=True)
                    detail = reward_details(text, tasks[index]["answer"], settings.get("reward_parser", "strict"))
                    record.update(detail, response_text=text)
                    return detail["reward"]
                records, metrics = run_iteration(engine, trainer, prompts, reward_fn, iteration=iteration,
                                                 max_tokens=settings["max_tokens"], sampling_config=metadata,
                                                 trajectory_callback=lambda rows: save_json(args.output_dir / f"rollouts-{iteration}.json", rows),
                                                 sync_probe=sync_probe)
                metrics["iteration"] = iteration
                save_json(args.output_dir / f"rollouts-{iteration}.json", records)
                metrics["parse_status"] = [r["parse_status"] for r in records]
                metrics["parse_failure_rate"] = sum(r["parse_status"] != "ok" for r in records) / len(records)
                lengths = [len(r["response_ids"]) for r in records]
                metrics["response_lengths"] = lengths
                metrics["response_length_mean"] = sum(lengths) / len(lengths)
                metrics["response_length_min"], metrics["response_length_max"] = min(lengths), max(lengths)
                metrics["reward_min"], metrics["reward_max"] = min(metrics["rewards"]), max(metrics["rewards"])
                metrics["mean_reward"] = sum(metrics["rewards"]) / len(records)
                metrics["truncation_fraction"] = sum(r["finish_reason"] == "length" for r in records) / len(records)
                if (iteration + 1) % settings["evaluation_every"] == 0:
                    metrics["validation"] = evaluate(engine, validation, label=str(iteration), max_tokens=settings["max_tokens"],
                                                     reward_parser=settings.get("reward_parser", "strict"))
                stream.write(json.dumps(metrics) + "\n"); stream.flush()
                summary["optimizer_steps"] += sum(step["optimizer_step"] for step in metrics["steps"])
                summary["policy_commits"] += int(metrics["updated"])
                change = (metrics.get("weight_update") or {}).get("weight_changes") or {}
                summary["actor_visible_commits"] += int(change.get("changed_fraction", 0) > 0)
                summary["probe_visible_commits"] += int(metrics.get("sync_probe", {}).get("max_absolute_change", 0) > 0)
                summary["skipped_batches"] += int(not metrics["updated"])
                latest = metrics["steps"][-1] if metrics["steps"] else {}
                flat = {"iteration": iteration, "mean_reward": metrics["mean_reward"],
                        **{key: metrics[key] for key in ("reward_min", "reward_max", "parse_failure_rate",
                            "response_length_mean", "response_length_min", "response_length_max")},
                        "loss": latest.get("loss"), "ratio_mean": latest.get("current_anchor_ratio_mean"),
                        "ratio_min": latest.get("ratio_min"), "ratio_max": latest.get("ratio_max"),
                        "actor_changed_fraction": (metrics.get("weight_update") or {}).get("weight_changes", {}).get("changed_fraction"),
                        "probe_max_absolute_change": metrics.get("sync_probe", {}).get("max_absolute_change"),
                        "validation_accuracy": metrics.get("validation", {}).get("accuracy"),
                        "policy_version": metrics["new_policy_version"], "updated": metrics["updated"],
                        "ratio_outside_fraction": latest.get("ratio_outside_fraction"),
                        "clip_active_fraction": latest.get("clip_active_fraction"),
                        "truncation_fraction": metrics["truncation_fraction"], **metrics["timings"]}
                if writer is None:
                    writer = csv.DictWriter(csv_stream, fieldnames=list(flat)); writer.writeheader()
                writer.writerow(flat); csv_stream.flush()
                print(json.dumps(metrics))
        summary["status"] = "completed_with_updates" if summary["optimizer_steps"] else "completed_no_learning_signal"
        save_json(args.output_dir / "summary.json", summary)
        if settings.get("save_final_checkpoint", False):
            from lockstep_rl.checkpoint import save_actor_checkpoint
            save_actor_checkpoint(model, engine.config.hf_config, engine.tokenizer,
                                  args.output_dir / "checkpoint-final", policy_version=engine.policy_version,
                                  eos_token_ids=engine.config.eos_token_ids)
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "engine_state": engine.state,
                   "policy_version": engine.policy_version}
        try:
            save_json(args.output_dir / "failure.json", failure)
        except Exception as log_error:
            print(f"Could not save failure.json: {log_error}", file=sys.stderr)
        raise
    finally:
        atexit.unregister(engine.exit)
        engine.exit()
    run_checked([sys.executable, "scripts/plot_results.py", "--kind", "training", "--input",
                    str(args.output_dir / "metrics.jsonl"), "--output", str(args.output_dir / "training.png")])


if __name__ == "__main__":
    main()
