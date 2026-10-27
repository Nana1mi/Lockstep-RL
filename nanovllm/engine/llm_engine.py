import atexit
import threading
from functools import wraps
from dataclasses import fields, replace
from time import perf_counter
from tqdm.auto import tqdm

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._lock = threading.RLock()
        from nanovllm.config import Config, load_eos_token_ids
        from transformers import AutoTokenizer
        import torch.multiprocessing as mp
        import torch.distributed as dist
        from nanovllm.engine.model_runner import ModelRunner
        config_fields = {field.name for field in fields(Config)}
        unknown = set(kwargs) - config_fields
        if unknown:
            raise ValueError(f"unknown engine configuration fields: {sorted(unknown)}")
        if dist.is_initialized():
            raise RuntimeError("an existing process group is active; use one engine per process")
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # Validate CPU assets before creating CUDA/NCCL state or changing the
        # Sequence block-size convention shared by this process.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, local_files_only=True)
        config.eos = self.tokenizer.eos_token_id
        config.eos_token_ids = load_eos_token_ids(config.model, config.eos)
        if any(token >= config.hf_config.vocab_size for token in config.eos_token_ids):
            raise ValueError("EOS ID outside model vocabulary")
        self.config = config
        self.policy_version = 0
        self.state = "ready"
        self._request_ids = set()
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    @synchronized
    def exit(self):
        if self.state == "closed":
            return
        self.state = "closed"
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def _require_ready(self):
        if self.state != "ready":
            raise RuntimeError(f"engine is {self.state}; reload a known checkpoint before generating")

    @synchronized
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, *,
                    request_id=None, group_id=None, forced_response_ids=None, sampling_config=None):
        seq = self._build_request(prompt, sampling_params, request_id=request_id, group_id=group_id,
                                  forced_response_ids=forced_response_ids, sampling_config=sampling_config)
        self.scheduler.add(seq)
        self._request_ids.add(seq.request_id)
        return seq.request_id

    def _build_request(self, prompt, sampling_params, *, request_id=None, group_id=None,
                       forced_response_ids=None, sampling_config=None):
        self._require_ready()
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if forced_response_ids is not None:
            sampling_params = replace(sampling_params, max_tokens=len(forced_response_ids))
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + response budget exceeds max_model_len")
        vocab = self.config.hf_config.vocab_size
        eos_ids = list(getattr(self.config, "eos_token_ids", ()) or ([self.config.eos] if self.config.eos is not None else []))
        all_ids = list(prompt) + (list(forced_response_ids) if forced_response_ids is not None else [])
        if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 or t >= vocab for t in all_ids):
            raise ValueError("token ID outside model vocabulary")
        metadata = {"model": self.config.model, "tokenizer": self.config.model,
                    "model_revision": getattr(self.config.hf_config, "_commit_hash", None),
                    "chat_template_version": "caller_supplied_token_ids", "enable_thinking": False,
                    "eos_token_id": self.config.eos, "eos_token_ids": eos_ids, "eos_rule": "any_configured_eos",
                    "top_k": None, "top_p": 1.0}
        reserved = {"model": self.config.model, "tokenizer": self.config.model,
                    "eos_token_id": self.config.eos, "eos_token_ids": eos_ids, "eos_rule": "any_configured_eos",
                    "enable_prefix_caching": getattr(self.config, "enable_prefix_caching", True),
                    "fp32_lm_head": getattr(self.config, "fp32_lm_head", False),
                    "cache_fp32_lm_head": getattr(self.config, "cache_fp32_lm_head", False),
                    "temperature": 1, "enable_thinking": False, "top_k": None, "top_p": 1.0}
        if any(key in (sampling_config or {}) and sampling_config[key] != value for key, value in reserved.items()):
            raise ValueError("sampling metadata conflicts with the first-version distribution/configuration")
        metadata.update(sampling_config or {})
        metadata.update(reserved)
        seq = Sequence(prompt, sampling_params, request_id=request_id, group_id=group_id,
                       policy_version=self.policy_version, forced_response_ids=forced_response_ids,
                       sampling_config=metadata)
        if seq.request_id in self._request_ids:
            raise ValueError("request_id must be unique within this engine")
        self.scheduler.validate_request(seq)
        return seq

    @synchronized
    def step(self, *, return_rollouts=False):
        self._require_ready()
        if self.is_finished():
            return [], 0
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids, logprobs = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler._record("execution", **getattr(self.model_runner, "last_execution", {}))
        self.scheduler.postprocess(seqs, token_ids, is_prefill, logprobs)
        outputs = [(seq.seq_id, seq.rollout().to_dict() if return_rollouts else seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    @synchronized
    def update_weights(self, state_dict, *, policy_version=None, measure_changes=False):
        self._require_ready()
        if self.config.tensor_parallel_size != 1:
            raise NotImplementedError("online updates are validated for TP=1 only")
        if not self.is_finished():
            raise RuntimeError("weight update requires an idle scheduler")
        version = self.policy_version + 1 if policy_version is None else policy_version
        if not isinstance(version, int) or isinstance(version, bool) or version <= self.policy_version:
            raise ValueError("policy_version must increase after a successful commit")
        # Preflight rejection leaves the current version fully serviceable.
        start = perf_counter()
        plan = self.model_runner.call("prepare_weight_update", state_dict)
        preflight_done = perf_counter()
        changes = self.model_runner.call("weight_change_statistics", plan) if measure_changes else None
        measurement_done = perf_counter()
        self.state = "updating"
        try:
            self.model_runner.call("commit_weight_update", plan)
            commit_done = perf_counter()
            self.scheduler.block_manager.invalidate()
            invalidate_done = perf_counter()
        except Exception:
            self.state = "broken"
            raise
        self.policy_version = version
        self.state = "ready"
        timings = {"preflight_seconds": preflight_done - start, "commit_seconds": commit_done - measurement_done,
                   "invalidate_seconds": invalidate_done - commit_done, "total_seconds": invalidate_done - start,
                   "change_measurement_seconds": measurement_done - preflight_done}
        timings["sync_seconds"] = sum(timings[key] for key in ("preflight_seconds", "commit_seconds", "invalidate_seconds"))
        self.scheduler._record("weight_commit", policy_version=version,
                               old_cache_entries=len(self.scheduler.block_manager.hash_to_block_id), **timings)
        return {"policy_version": version, "timings": timings, "weight_changes": changes}

    @synchronized
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        *, request_ids=None, group_ids=None, forced_response_ids=None, sampling_config=None,
    ) -> list[dict]:
        self._require_ready()
        if not self.is_finished():
            raise RuntimeError("generate requires an idle scheduler; drain add_request/step first")
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        values = [sampling_params, request_ids, group_ids, forced_response_ids]
        if any(value is not None and len(value) != len(prompts) for value in values):
            raise ValueError("all per-request arguments must match the prompt count")
        pending = [self._build_request(
            prompt, sp, request_id=request_ids[i] if request_ids else None,
            group_id=group_ids[i] if group_ids else None,
            forced_response_ids=forced_response_ids[i] if forced_response_ids else None,
            sampling_config=sampling_config) for i, (prompt, sp) in enumerate(zip(prompts, sampling_params))]
        if len({seq.request_id for seq in pending}) != len(pending):
            raise ValueError("request IDs in a generation batch must be unique")
        for seq in pending:
            self.scheduler.add(seq)
            self._request_ids.add(seq.request_id)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step(return_rollouts=True)
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{**record, "text": self.tokenizer.decode(record["response_ids"]),
                    "token_ids": record["response_ids"]} for record in outputs]
        return outputs
