"""Synchronous full-parameter GRPO updates and rollout/sync loop integration."""

from contextlib import nullcontext
import time

import torch

from lockstep_rl.batch import collate_rollouts
from lockstep_rl.grpo import GRPOConfig, accumulation_count, clipped_surrogate, group_advantages
from lockstep_rl.logprobs import selected_logprobs


def causal_logprobs(model, batch, *, chunk_size, fp32_lm_head=False):
    positions = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
    hidden = model.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                         position_ids=positions, use_cache=False).last_hidden_state
    # Prompt/padding positions never enter the vocabulary projection. Scatter
    # keeps the existing dense score/mask contract and its differentiable shift.
    mask = batch.response_mask
    response_hidden = hidden[:, :-1][mask]
    targets = batch.input_ids[:, 1:][mask]
    if not targets.numel():
        return hidden[:, :-1].sum(-1).float() * 0
    selected = selected_logprobs(response_hidden, model.get_output_embeddings(), targets,
                                chunk_size, fp32_lm_head=fp32_lm_head)
    return selected.new_zeros(mask.shape).masked_scatter(mask, selected)


@torch.no_grad()
def weight_cast_statistics(model, target_dtype):
    total = maximum = None
    count = 0
    for parameter in model.parameters():
        for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
            error = (chunk.to(target_dtype).to(chunk.dtype).float() - chunk.float()).abs()
            value, peak = error.sum(), error.max()
            total = value if total is None else total + value
            maximum = peak if maximum is None else torch.maximum(maximum, peak)
            count += chunk.numel()
    if not count:
        raise ValueError("cannot measure an empty model")
    return {"target_dtype": str(target_dtype), "parameter_values": count,
            "mean_absolute_error": float(total / count), "max_absolute_error": float(maximum)}


class GRPOTrainer:
    def __init__(self, model, optimizer, *, config=None, pad_token_id=0, score_fn=causal_logprobs):
        self.model = model
        self.optimizer = optimizer
        self.config = config or GRPOConfig()
        self.pad_token_id = pad_token_id
        self.score_fn = score_fn

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.config.autocast_bf16 else nullcontext()

    def _score(self, batch):
        with self._autocast():
            options = {"fp32_lm_head": True} if self.config.fp32_lm_head else {}
            return self.score_fn(self.model, batch, chunk_size=self.config.logprob_chunk_size, **options)

    def update(self, records, rewards):
        if not records:
            raise ValueError("empty rollout batch")
        if any(r.get("sampling_config", {}).get("diagnostic_forced", False) or r.get("diagnostic_scoring", False)
               for r in records):
            raise ValueError("forced diagnostic trajectories are not sampled RL training data")
        if len({r["policy_version"] for r in records}) != 1:
            raise ValueError("synchronous training requires a single rollout policy version")
        prompts_by_group = {}
        for record in records:
            prompt = tuple(record["prompt_ids"])
            previous = prompts_by_group.setdefault(record["group_id"], prompt)
            if previous != prompt:
                raise ValueError("responses in one group must have identical prompt tokens")
        accumulation_count(len(records), self.config)
        advantages, group_stats = group_advantages(rewards, [r["group_id"] for r in records],
                                                   group_size=self.config.group_size,
                                                   epsilon=self.config.advantage_epsilon)
        total_tokens = sum(sum(r["loss_mask"]) for r in records)
        result = {"updated": False, "steps": [], "group_stats": group_stats,
                  "valid_tokens": total_tokens, "anchor": self.config.anchor,
                  "policy_version": records[0]["policy_version"], "beta": 0,
                  "advantage_values": advantages.tolist()}
        if total_tokens == 0 or not advantages.count_nonzero():
            self.optimizer.zero_grad(set_to_none=True)
            result["skip_reason"] = "empty_mask" if total_tokens == 0 else "zero_advantages"
            return result
        # HF activation checkpointing is gated on training mode. Disable dropout
        # modules individually so checkpointing works without stochastic ratios.
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                                   torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)):
                module.eval()
            if getattr(module, "attention_dropout", 0) != 0:
                raise ValueError("functional attention dropout must be configured to zero")
        microbatches = []
        for start in range(0, len(records), self.config.microbatch_size):
            batch = collate_rollouts(records[start:start + self.config.microbatch_size],
                                     pad_token_id=self.pad_token_id, device=self.device)
            advantage = advantages[start:start + self.config.microbatch_size].to(self.device)
            with torch.no_grad():
                initial = self._score(batch).detach()
                anchor = initial.clone() if self.config.anchor == "recomputed" else batch.rollout_logprobs.detach().clone()
                delta = (initial - batch.rollout_logprobs)[batch.response_mask]
            microbatches.append((batch, advantage, anchor, delta.cpu()))
        result["pre_update_current_minus_rollout"] = torch.cat([x[3] for x in microbatches]).tolist()
        result["pre_update_comparison_scope"] = "includes backend and any FP32-to-BF16 weight-cast differences"
        result["frozen_anchors"] = [anchor[batch.response_mask].detach().cpu().tolist()
                                    for batch, _, anchor, _ in microbatches]
        for update in range(self.config.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            counters = []
            for batch, advantage, anchor, _ in microbatches:
                current = self._score(batch)
                loss, stats = clipped_surrogate(current, anchor, advantage, batch.response_mask,
                                                epsilon=self.config.clip_epsilon,
                                                total_valid_tokens=total_tokens)
                loss.backward()
                loss_value += float(loss.detach())
                counters.append(stats)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                                                       error_if_nonfinite=True)
            n = sum(s["valid_tokens"] for s in counters)
            step_metrics = {"index": update, "loss": loss_value, "grad_norm": float(grad_norm),
                            "current_anchor_ratio_mean": sum(s["ratio_sum"] for s in counters) / n,
                            "ratio_outside_fraction": sum(s["ratio_outside"] for s in counters) / n,
                            "clip_active_fraction": sum(s["clip_active"] for s in counters) / n,
                            "ratio_min": min(s["ratio_min"] for s in counters if s["valid_tokens"]),
                            "ratio_max": max(s["ratio_max"] for s in counters if s["valid_tokens"])}
            if grad_norm <= 0:
                result["skip_reason"] = "zero_gradient"
                step_metrics.update(optimizer_step=False, max_observed_parameter_delta=0.)
                result["steps"].append(step_metrics)
                break
            observation = self._observe_nonzero_gradient()
            self.optimizer.step()
            change = max(float((parameter.detach().reshape(-1)[start:start + len(before)] - before).abs().max())
                         for parameter, start, before in observation)
            if change == 0:
                raise RuntimeError("optimizer did not change the observed nonzero-gradient parameter")
            step_metrics.update(optimizer_step=True, max_observed_parameter_delta=change)
            result["steps"].append(step_metrics)
            result["updated"] = True
        self.optimizer.zero_grad(set_to_none=True)
        return result

    def _observe_nonzero_gradient(self):
        parameters = sorted((p for p in self.model.parameters() if p.grad is not None), key=lambda p: p.numel())
        observations = []
        for p in parameters:
            flat_grad = p.grad.detach().reshape(-1)
            for start in range(0, flat_grad.numel(), 4096):
                if flat_grad[start:start + 4096].count_nonzero():
                    observations.append((p, start, p.detach().reshape(-1)[start:start + 4096].clone()))
                    break
            if len(observations) == 8:
                return observations
        if observations:
            return observations
        raise RuntimeError("no nonzero gradient found despite positive global norm")


def run_iteration(engine, trainer, prompts, reward_fn, *, iteration, max_tokens=128, sampling_config=None,
                  trajectory_callback=None, sync_probe=None):
    from nanovllm import SamplingParams
    size = trainer.config.group_size
    expanded = [prompt for prompt in prompts for _ in range(size)]
    groups = [f"{iteration}:{i}" for i in range(len(prompts)) for _ in range(size)]
    requests = [f"{iteration}:{i}:{j}" for i in range(len(prompts)) for j in range(size)]
    def stamp():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()
    start = stamp()
    records = engine.generate(expanded, SamplingParams(max_tokens=max_tokens), request_ids=requests,
                              group_ids=groups, sampling_config=sampling_config, use_tqdm=False)
    if trajectory_callback is not None:
        trajectory_callback(records)
    after_rollout = stamp()
    if any(r["policy_version"] != engine.policy_version for r in records):
        raise RuntimeError("rollout records do not match the committed engine version")
    prompt_by_group = {f"{iteration}:{i}": i for i in range(len(prompts))}
    if len(records) != len(requests) or {r["request_id"] for r in records} != set(requests):
        raise RuntimeError("rollout request identity/count mismatch")
    rewards = [reward_fn(record, prompt_by_group[record["group_id"]]) for record in records]
    after_reward = stamp()
    cast_error = weight_cast_statistics(trainer.model, engine.config.hf_config.dtype)
    after_cast_diagnostics = stamp()
    metrics = trainer.update(records, rewards)
    after_training = stamp()
    old_version = engine.policy_version
    sync_result = None
    def probe_scores(label):
        from nanovllm import SamplingParams
        return engine.generate([sync_probe["prompt_ids"]], SamplingParams(ignore_eos=True),
            forced_response_ids=[sync_probe["response_ids"]], request_ids=[f"sync-probe:{iteration}:{label}"],
            use_tqdm=False)[0]["response_logprobs"]
    before_probe = probe_scores("before") if sync_probe and metrics["updated"] else None
    before_sync = stamp()
    if metrics["updated"]:
        sync_result = engine.update_weights(trainer.model.state_dict(), measure_changes=sync_probe is not None)
    after_sync = stamp()
    after_probe = probe_scores("after") if before_probe is not None else None
    after_probe_time = stamp()
    if before_probe is not None:
        delta = [new - old for old, new in zip(before_probe, after_probe)]
        metrics["sync_probe"] = {"prompt_ids": sync_probe["prompt_ids"], "response_ids": sync_probe["response_ids"],
            "before_logprobs": before_probe, "after_logprobs": after_probe, "delta_logprobs": delta,
            "max_absolute_change": max(abs(value) for value in delta),
            "mean_absolute_change": sum(abs(value) for value in delta) / len(delta)}
    metrics.update(rewards=rewards, old_policy_version=old_version, new_policy_version=engine.policy_version,
                   weight_update=sync_result,
                   rollout_includes_persistence=trajectory_callback is not None,
                   pre_update_weight_cast_error=cast_error,
                   timings={"rollout": after_rollout - start, "reward": after_reward - after_rollout,
                            "weight_cast_diagnostics": after_cast_diagnostics - after_reward,
                            "training": after_training - after_cast_diagnostics,
                            "sync": sync_result["timings"]["sync_seconds"] if sync_result else 0.,
                            "sync_wall_including_measurement": after_sync - before_sync,
                            "sync_probe": before_sync - after_training + after_probe_time - after_sync,
                            "total": after_probe_time - start})
    return records, metrics
