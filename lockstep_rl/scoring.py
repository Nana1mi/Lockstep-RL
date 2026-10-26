"""Teacher-forced scoring of the exact rollout token trajectory."""

from copy import deepcopy

import torch

from lockstep_rl.batch import collate_rollouts
from lockstep_rl.trainer import causal_logprobs


@torch.no_grad()
def score_records(model, records, *, pad_token_id=0, chunk_size=32, fp32_lm_head=False):
    """Score one response at a time to keep large-vocabulary memory bounded."""
    model.eval()
    device = next(model.parameters()).device
    output = []
    for record in records:
        if any(value != 1 for value in record["loss_mask"]):
            raise ValueError("diagnostics require complete unfiltered rollout tokens")
        batch = collate_rollouts([record], pad_token_id=pad_token_id, device=device)
        scores = causal_logprobs(model, batch, chunk_size=chunk_size, fp32_lm_head=fp32_lm_head)
        result = deepcopy(record)
        result["response_logprobs"] = scores[batch.response_mask].float().cpu().tolist()
        result["diagnostic_scoring"] = True
        result["scoring_dtype"] = str(next(model.parameters()).dtype)
        result["fp32_lm_head"] = fp32_lm_head
        output.append(result)
    return output


def load_effective_hf(path, *, dtype, device, attention_backend="sdpa"):
    """FP32 reference promotes the same BF16 effective weights, not original FP32."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                attn_implementation=attention_backend,
                                                local_files_only=True)
    return model.to(device=device, dtype=dtype).eval()
