from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams
from nanovllm.rollout import Rollout


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=None, *, request_id=None,
                 group_id=None, policy_version=0, sampling_config=None, forced_response_ids=None):
        sampling_params = sampling_params or SamplingParams()
        if not token_ids or any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in token_ids):
            raise ValueError("prompt must contain nonnegative integer token IDs")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.request_id = str(self.seq_id) if request_id is None else str(request_id)
        self.group_id = self.request_id if group_id is None else str(group_id)
        if not self.request_id or not self.group_id:
            raise ValueError("request_id and group_id cannot be empty")
        self.policy_version = policy_version
        self.response_logprobs = []
        self.loss_mask = []
        self.finish_reason = None
        self.forced_response_ids = tuple(forced_response_ids) if forced_response_ids is not None else None
        if self.forced_response_ids is not None:
            if not self.forced_response_ids or len(self.forced_response_ids) != self.max_tokens:
                raise ValueError("forced response length must match max_tokens")
            if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in self.forced_response_ids):
                raise ValueError("forced IDs must be nonnegative integers")
        self.sampling_config = dict(sampling_config or {})
        self.sampling_config.update(temperature=self.temperature, max_tokens=self.max_tokens,
                                    ignore_eos=self.ignore_eos,
                                    diagnostic_forced=self.forced_response_ids is not None)

    @property
    def next_forced_token(self):
        if self.forced_response_ids is None:
            return -1
        return self.forced_response_ids[self.num_completion_tokens]

    def rollout(self):
        if not self.is_finished:
            raise RuntimeError("only finished sequences have complete rollout records")
        return Rollout(self.request_id, self.group_id, self.prompt_token_ids,
                       self.completion_token_ids, list(self.response_logprobs),
                       list(self.loss_mask), self.finish_reason, self.policy_version,
                       dict(self.sampling_config))

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
