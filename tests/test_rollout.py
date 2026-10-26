import threading
import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import sample_with_logprobs
from nanovllm.sampling_params import SamplingParams


def cpu_engine(blocks=16, budget=3):
    engine = LLMEngine.__new__(LLMEngine)
    engine._lock = threading.RLock()
    engine.state = "ready"
    engine.policy_version = 0
    engine._request_ids = set()
    engine.config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=budget,
                                   eos=99, kvcache_block_size=4, num_kvcache_blocks=blocks,
                                   tensor_parallel_size=1, max_model_len=64, model="tiny-cpu",
                                   hf_config=SimpleNamespace(vocab_size=100, dtype=torch.float32))
    engine.scheduler = Scheduler(engine.config)
    engine.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
    class Runner:
        def call(self, method, seqs, is_prefill):
            return ([seq.next_forced_token if seq.forced_response_ids else 7 for seq in seqs],
                    [-0.75 for _ in seqs])
    engine.model_runner = Runner()
    return engine


class RolloutTest(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_sampler_is_fp32_correct_and_does_not_modify_inputs(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 31, dtype=torch.bfloat16)
        original = logits.clone()
        tokens, scores = sample_with_logprobs(logits, torch.ones(4))
        torch.testing.assert_close(scores, logits.float().log_softmax(-1).gather(1, tokens[:, None]).flatten())
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(logits, original)
        forced = torch.tensor([2, -1, 9, 30])
        tokens, scores = sample_with_logprobs(logits, torch.ones(4), forced)
        self.assertEqual(tokens[[0, 2, 3]].tolist(), [2, 9, 30])
        torch.testing.assert_close(scores, logits.float().log_softmax(-1).gather(1, tokens[:, None]).flatten())

    def test_chunked_prefill_does_not_advance_replay(self):
        engine = cpu_engine()
        engine.add_request(list(range(10)), SamplingParams(), request_id="r", group_id="g",
                           forced_response_ids=[21, 22])
        seq = engine.scheduler.waiting[0]
        for _ in range(3):
            self.assertEqual(engine.step()[0], [])
            self.assertEqual(seq.num_completion_tokens, 0)
            self.assertEqual(seq.response_logprobs, [])
            self.assertEqual(seq.next_forced_token, 21)
        self.assertEqual(engine.step()[0], [])
        self.assertEqual(seq.num_completion_tokens, 1)
        result, _ = engine.step(return_rollouts=True)
        record = result[0][1]
        self.assertEqual(record["response_ids"], [21, 22])
        self.assertEqual(record["response_logprobs"], [-0.75, -0.75])
        self.assertEqual(record["loss_mask"], [1, 1])
        self.assertEqual(record["finish_reason"], "length")
        self.assertEqual(record["request_id"], "r")
        self.assertEqual(record["group_id"], "g")
        self.assertTrue(record["sampling_config"]["diagnostic_forced"])

    def test_eos_is_retained_and_ignore_eos_means_length(self):
        for ignore, reason in [(False, "eos"), (True, "length")]:
            engine = cpu_engine(budget=16)
            records = engine.generate([[1, 2]], SamplingParams(ignore_eos=ignore),
                                      forced_response_ids=[[99]], use_tqdm=False)
            self.assertEqual(records[0]["response_ids"], [99])
            self.assertEqual(records[0]["finish_reason"], reason)
            self.assertEqual(records[0]["sampling_config"]["eos_token_id"], 99)
            self.assertEqual(records[0]["sampling_config"]["eos_rule"], "any_configured_eos")

    def test_either_generation_config_eos_stops_and_list_is_recorded(self):
        for token in (98, 99):
            engine = cpu_engine(budget=16)
            engine.config.eos_token_ids = (98, 99)
            engine.scheduler.eos_token_ids = {98, 99}
            row = engine.generate([[1, 2]], SamplingParams(max_tokens=3),
                forced_response_ids=[[token, 7, 8]], use_tqdm=False)[0]
            self.assertEqual(row["response_ids"], [token])
            self.assertEqual(row["finish_reason"], "eos")
            self.assertEqual(row["sampling_config"]["eos_token_ids"], [98, 99])

    def test_preemption_replays_context_without_rewriting_old_scores(self):
        engine = cpu_engine(budget=16)
        engine.add_request([1, 2, 3, 4, 5], SamplingParams(), forced_response_ids=[21, 22])
        engine.step()
        seq = engine.scheduler.running.pop()
        engine.scheduler.preempt(seq)
        self.assertEqual(seq.next_forced_token, 22)
        result, _ = engine.step(return_rollouts=True)
        self.assertEqual(result[0][1]["response_ids"], [21, 22])
        self.assertEqual(result[0][1]["response_logprobs"], [-0.75, -0.75])

    def test_prefix_hit_and_logical_invalidation(self):
        engine = cpu_engine(budget=16)
        engine.generate([list(range(10))], SamplingParams(max_tokens=1), use_tqdm=False)
        manager = engine.scheduler.block_manager
        seq = Sequence(list(range(10)), SamplingParams(max_tokens=1))
        self.assertEqual(manager.can_allocate(seq), 2)
        manager.invalidate()
        self.assertEqual(manager.can_allocate(seq), 0)
        self.assertFalse(manager.hash_to_block_id)

    def test_bad_batch_admission_does_not_queue_partial_work(self):
        engine = cpu_engine()
        for kwargs in ({"request_ids": ["same", "same"]},
                       {"forced_response_ids": [[1], [-1]]},
                       {"group_ids": ["one"]}):
            with self.assertRaises(ValueError):
                engine.generate([[1], [2]], SamplingParams(max_tokens=2), use_tqdm=False, **kwargs)
            self.assertTrue(engine.is_finished())
        with self.assertRaises(ValueError):
            cpu_engine(blocks=1).add_request([1, 2], SamplingParams(max_tokens=10))
        with self.assertRaises(ValueError):
            SamplingParams(temperature=0.6)
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)


if __name__ == "__main__":
    unittest.main()
