import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        fp32_lm_head: bool = False,
        cache_fp32_lm_head: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.fp32_lm_head = fp32_lm_head
        self.cache_fp32_lm_head = cache_fp32_lm_head
        self.register_buffer("_fp32_weight", None, persistent=False)

    @torch.no_grad()
    def refresh_fp32_weight(self):
        """Initial full load / successful commit only; never a forward side effect."""
        if not self.cache_fp32_lm_head:
            return
        if not self.fp32_lm_head:
            raise ValueError("cached FP32 head requires fp32_lm_head")
        if self._fp32_weight is None:
            self._fp32_weight = self.weight.detach().to(dtype=torch.float32, copy=True)
        else:
            if self._fp32_weight.dtype != torch.float32 or self._fp32_weight.device != self.weight.device:
                raise RuntimeError("cached head dtype/device changed; reinitialize the engine")
            self._fp32_weight.copy_(self.weight.detach())

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        if self.fp32_lm_head:
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.cache_fp32_lm_head:
                    if self._fp32_weight is None:
                        raise RuntimeError("cached FP32 head was not initialized by a full load/commit")
                    weight = self._fp32_weight
                else:
                    weight = self.weight.float()
                logits = F.linear(x.float(), weight)
        else:
            logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
