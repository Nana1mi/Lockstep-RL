import torch
from torch import nn


def sample_with_logprobs(logits, temperatures, forced_token_ids=None):
    """Return sampled/forced IDs and their FP32 scores without mutating logits."""
    scaled = logits.float() / temperatures.unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    token_ids = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
    if forced_token_ids is not None:
        token_ids = torch.where(forced_token_ids >= 0, forced_token_ids, token_ids)
    scores = scaled.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - scaled.logsumexp(-1)
    return token_ids, scores


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, forced_token_ids=None):
        return sample_with_logprobs(logits, temperatures, forced_token_ids)
