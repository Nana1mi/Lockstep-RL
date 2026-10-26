"""The PLAN.md token-normalized, reference-free GRPO variant (beta=0)."""

from collections import defaultdict
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-6
    anchor: str = "rollout"
    update_steps: int = 1
    microbatch_size: int = 1
    gradient_accumulation_steps: int | None = None
    logprob_chunk_size: int = 32
    max_grad_norm: float = 1.0
    autocast_bf16: bool = False

    def __post_init__(self):
        for name in ("group_size", "update_steps", "microbatch_size", "logprob_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if not all(math.isfinite(value) for value in (self.clip_epsilon, self.advantage_epsilon, self.max_grad_norm)):
            raise ValueError("GRPO numerical settings must be finite")
        if self.group_size < 2 or self.anchor not in ("rollout", "recomputed"):
            raise ValueError("group_size >= 2 and a supported clipping anchor are required")
        if not 0 < self.clip_epsilon < 1 or self.advantage_epsilon <= 0:
            raise ValueError("invalid clipping/advantage epsilon")
        if min(self.update_steps, self.microbatch_size, self.logprob_chunk_size) < 1 or self.max_grad_norm <= 0:
            raise ValueError("update/microbatch/chunk budgets and max_grad_norm must be positive")
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive or None (auto)")


def group_advantages(rewards, group_ids, *, group_size=4, epsilon=1e-6):
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    if rewards.ndim != 1 or len(group_ids) != len(rewards) or not torch.isfinite(rewards).all():
        raise ValueError("finite rewards and aligned group IDs are required")
    if group_size < 2 or epsilon <= 0:
        raise ValueError("invalid group size or epsilon")
    groups = defaultdict(list)
    for i, group in enumerate(group_ids):
        groups[group].append(i)
    advantages = torch.zeros_like(rewards)
    zero_variance = 0
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError("each explicit group must contain exactly group_size samples")
        values = rewards[indices]
        std = values.std(correction=0)
        if std == 0:
            zero_variance += 1
        else:
            advantages[indices] = (values - values.mean()) / (std + epsilon)
    return advantages, {"groups": len(groups), "zero_variance_groups": zero_variance}


def clipped_surrogate(current, anchor, advantages, mask, *, epsilon=0.2, total_valid_tokens=None):
    """Return a globally token-normalized loss and additive diagnostic counters."""
    if current.shape != anchor.shape or current.shape != mask.shape or current.ndim != 2:
        raise ValueError("current, anchor and mask must have the same [batch, token] shape")
    if advantages.shape != (current.shape[0],):
        raise ValueError("advantages must have one scalar per response")
    if not 0 < epsilon < 1 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("invalid epsilon or nonbinary mask")
    valid = mask.bool()
    n = int(valid.sum().item())
    denominator = n if total_valid_tokens is None else total_valid_tokens
    if denominator < n or denominator < 0:
        raise ValueError("global token denominator cannot be smaller than the microbatch")
    if n == 0:
        return current[valid].sum() * 0, {"valid_tokens": 0, "ratio_outside": 0, "clip_active": 0,
                                         "ratio_sum": 0., "ratio_min": None, "ratio_max": None}
    c = current[valid].float()
    a = anchor.detach()[valid].float()
    advantage = advantages.detach()[:, None].expand_as(current)[valid].float()
    if not torch.isfinite(c).all() or not torch.isfinite(a).all() or not torch.isfinite(advantage).all():
        raise FloatingPointError("nonfinite valid-token logprobs or advantages")
    ratio = torch.exp(c - a)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("importance ratio overflow")
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    loss = -torch.minimum(unclipped, clipped).sum() / denominator
    with torch.no_grad():
        outside = (ratio < 1 - epsilon) | (ratio > 1 + epsilon)
        active = ((advantage > 0) & (ratio > 1 + epsilon)) | ((advantage < 0) & (ratio < 1 - epsilon))
        stats = {"valid_tokens": n, "ratio_outside": int(outside.sum()), "clip_active": int(active.sum()),
                 "ratio_sum": float(ratio.sum()), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())}
    return loss, stats


def accumulation_count(sample_count, config):
    expected = math.ceil(sample_count / config.microbatch_size)
    if config.gradient_accumulation_steps is not None and config.gradient_accumulation_steps != expected:
        raise ValueError("gradient_accumulation_steps must cover this entire rollout batch exactly")
    return expected
