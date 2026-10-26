"""Right-padded teacher-forcing batches with explicit next-token response masks."""

from dataclasses import dataclass

import torch


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    rollout_logprobs: torch.Tensor


def collate_rollouts(records, *, pad_token_id, device="cpu"):
    if not records:
        raise ValueError("cannot collate an empty response list")
    lengths = [len(r["prompt_ids"]) + len(r["response_ids"]) for r in records]
    width = max(lengths)
    if width < 2:
        width = 2  # Empty masks can still be represented for explicit skip handling.
    ids = torch.full((len(records), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(records), width - 1), dtype=torch.bool, device=device)
    logprobs = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    for i, record in enumerate(records):
        prompt, response = record["prompt_ids"], record["response_ids"]
        if not prompt or len(response) != len(record["response_logprobs"]) or len(response) != len(record["loss_mask"]):
            raise ValueError("invalid prompt or misaligned response contract")
        if any(m not in (0, 1) for m in record["loss_mask"]):
            raise ValueError("training mask must be binary")
        ids[i, :lengths[i]] = torch.tensor(prompt + response, dtype=torch.long, device=device)
        attention[i, :lengths[i]] = 1
        start = len(prompt) - 1  # Hidden state at Lp-1 predicts the first response token.
        mask[i, start:start + len(response)] = torch.tensor(record["loss_mask"], dtype=torch.bool, device=device)
        logprobs[i, start:start + len(response)] = torch.tensor(record["response_logprobs"], dtype=torch.float32, device=device)
    return TokenBatch(ids, attention, mask, logprobs)
