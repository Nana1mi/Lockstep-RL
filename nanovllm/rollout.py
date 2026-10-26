"""Serializable rollout contract; does not import CUDA or model dependencies."""

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass
class Rollout:
    request_id: str
    group_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]
    loss_mask: list[int]
    finish_reason: str
    policy_version: int
    sampling_config: dict[str, Any]

    def __post_init__(self):
        n = len(self.response_ids)
        if not self.request_id or not self.group_id or not self.prompt_ids:
            raise ValueError("request/group IDs and prompt tokens must be nonempty")
        if len(self.response_logprobs) != n or len(self.loss_mask) != n:
            raise ValueError("response tokens, logprobs and mask must align")
        if any(not math.isfinite(x) or x > 1e-5 for x in self.response_logprobs):
            raise ValueError("logprobs must be finite log probabilities")
        if any(x != 1 for x in self.loss_mask):
            raise ValueError("unbatched rollout contains only actual tokens, all with mask=1")
        if self.finish_reason not in ("eos", "length") or self.policy_version < 0:
            raise ValueError("invalid finish reason or policy version")
        if self.sampling_config.get("temperature") != 1:
            raise ValueError("first-version rollout requires temperature=1")

    def to_dict(self):
        return asdict(self)
