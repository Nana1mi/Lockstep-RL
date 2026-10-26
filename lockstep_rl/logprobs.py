"""Selected-token scores with bounded vocabulary activation storage."""

import torch
from torch.utils.checkpoint import checkpoint


def selected_logprobs(hidden_states, output_head, target_ids, chunk_size=32):
    """Score aligned targets; caller is responsible for the causal shift.

    Input shapes are (..., hidden_size) and (...). The output is FP32, and
    gradients flow into both hidden states and the output head. Checkpointing
    recomputes vocabulary logits during backward instead of retaining all chunks.
    No temperature, top-k, top-p or other distribution transform is applied.
    """
    if hidden_states.shape[:-1] != target_ids.shape:
        raise ValueError("hidden states and target IDs must have aligned shapes")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if target_ids.dtype != torch.long:
        raise ValueError("target_ids must be int64")
    if hidden_states.device != target_ids.device:
        raise ValueError("hidden states and targets must share a device")
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = target_ids.reshape(-1)
    if flat_targets.numel() == 0:
        raise ValueError("at least one target is required")

    def score_chunk(hidden, targets):
        logits = output_head(hidden).float()
        chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return chosen - torch.logsumexp(logits, dim=-1)

    scores = []
    for start in range(0, flat_targets.numel(), chunk_size):
        hidden = flat_hidden[start:start + chunk_size]
        targets = flat_targets[start:start + chunk_size]
        if torch.is_grad_enabled():
            score = checkpoint(score_chunk, hidden, targets, use_reentrant=False)
        else:
            score = score_chunk(hidden, targets)
        scores.append(score)
    return torch.cat(scores).reshape(target_ids.shape)
