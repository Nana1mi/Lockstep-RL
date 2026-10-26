import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    record_trace: bool = False
    diagnostic_kv_blocks: int | None = None

    def __post_init__(self):
        for name in ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "tensor_parallel_size", "kvcache_block_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if not os.path.isdir(self.model):
            raise ValueError("model must be a local checkpoint directory")
        if not any(name.endswith(".safetensors") for name in os.listdir(self.model)):
            raise ValueError("model directory has no local safetensors weights")
        if self.kvcache_block_size < 256 or self.kvcache_block_size % 256:
            raise ValueError("KV block size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if min(self.max_num_seqs, self.max_num_batched_tokens, self.max_model_len) < 1:
            raise ValueError("scheduler and context budgets must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.diagnostic_kv_blocks is not None and (not isinstance(self.diagnostic_kv_blocks, int)
                                                     or isinstance(self.diagnostic_kv_blocks, bool)
                                                     or self.diagnostic_kv_blocks < 1):
            raise ValueError("diagnostic_kv_blocks must be positive")
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
