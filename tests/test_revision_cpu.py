"""Revision acceptance on CPU; cache simulations are not GPU measurements."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lockstep_rl.batch import collate_rollouts
from lockstep_rl.diagnostic_prompts import gsm8k_prompts
from lockstep_rl.trainer import causal_logprobs
from nanovllm.config import load_eos_token_ids
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils import weight_update as updates
from scripts.weight_sync_experiment import assert_outputs_close
from scripts.nano_path_worker import replay_groups
from tests.test_grpo import ToyLM, records
from tests.test_rollout import cpu_engine


class RevisionCPU(unittest.TestCase):
    def test_source_flags_are_one_reduction_per_tensor_and_one_host_batch(self):
        model = nn.Module()
        model.embed = nn.Embedding(1025, 4).to(torch.bfloat16)
        model.head = nn.Linear(4, 1025, bias=False).to(torch.bfloat16)
        model.head.weight = model.embed.weight
        source = torch.randn(1025, 4)
        state = {"embed.weight": source, "head.weight": source.detach()}
        with patch.object(updates, "_finite_after_conversion", wraps=updates._finite_after_conversion) as finite, \
             patch.object(updates, "_check_device_flags", wraps=updates._check_device_flags) as flags:
            plan = updates.prepare_weight_update(model, state)
        self.assertEqual(finite.call_count, 1)  # tied source is checked once, not 9 chunks
        self.assertEqual(flags.call_count, 1)
        self.assertTrue(all(flag.ndim == 0 for _, flag in flags.call_args.args[0]))
        with patch("torch.equal", side_effect=AssertionError("no embedding CPU equality copy")):
            self.assertIs(updates._equal_after_conversion(source, source.detach(), torch.bfloat16), True)
        self.assertEqual(len(plan), 2)

    def test_actor_change_fraction_measures_bf16_rounding_and_unique_storage(self):
        model = nn.Module()
        model.embed = nn.Embedding(3, 2).to(torch.bfloat16)
        model.head = nn.Linear(2, 3, bias=False).to(torch.bfloat16)
        model.head.weight = model.embed.weight
        with torch.no_grad():
            model.embed.weight.fill_(1)
        source = model.embed.weight.float() + 1e-6
        plan = updates.prepare_weight_update(model, {"embed.weight": source, "head.weight": source})
        self.assertEqual(updates.effective_weight_change_statistics(plan)["changed_fraction"], 0)
        source[0, 0] += .25
        stats = updates.effective_weight_change_statistics(plan)
        self.assertEqual(stats["total_values"], 6)
        self.assertEqual(stats["changed_values"], 1)
        self.assertAlmostEqual(stats["changed_fraction"], 1 / 6)
        updates.commit_weight_update(plan)
        self.assertEqual(float(model.embed.weight[0, 0].detach()), 1.25)

    def test_response_only_projection_matches_dense_values_and_gradients(self):
        torch.manual_seed(133)
        model = ToyLM()
        batch = collate_rollouts(records(), pad_token_id=0)
        hidden = model.model(batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        dense = model.head(hidden[:, :-1]).log_softmax(-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
        expected_grads = torch.autograd.grad(dense[batch.response_mask].sum(), tuple(model.parameters()))
        sizes = []
        handle = model.head.register_forward_pre_hook(lambda _, args: sizes.append(args[0].shape[0]))
        try:
            actual = causal_logprobs(model, batch, chunk_size=3)
        finally:
            handle.remove()
        self.assertEqual(sum(sizes), int(batch.response_mask.sum()))
        self.assertEqual(actual[~batch.response_mask].count_nonzero(), 0)
        actual_grads = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
        torch.testing.assert_close(actual[batch.response_mask], dense[batch.response_mask], atol=2e-6, rtol=2e-6)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_local_generation_eos_list_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_eos_token_ids(directory, 3), (3,))
            path = Path(directory) / "generation_config.json"
            path.write_text(json.dumps({"eos_token_id": [3, 4, 3]}))
            self.assertEqual(load_eos_token_ids(directory, 9), (3, 4))
            path.write_text(json.dumps({"eos_token_id": [True]}))
            with self.assertRaises(ValueError):
                load_eos_token_ids(directory, 9)

    def test_gsm_prompt_fixture_preserves_full_question_template_and_real_companions(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is False
                return "<user>" + messages[0]["content"] + "</user><assistant>"
            def encode(self, text, **kwargs):
                return list(text.encode())
        questions = [{"id": str(i), "prompt": f"Question number {i}: what is 2+3?", "answer": "5"} for i in range(5)]
        examples = [{"id": "example", "prompt": "Worked exercise", "answer": "8",
                     "solution": "Eight apples are shared, leaving eight. " * 30 + "#### 8"}]
        with tempfile.TemporaryDirectory() as directory:
            q, f = Path(directory) / "q.jsonl", Path(directory) / "f.jsonl"
            q.write_text("".join(json.dumps(row) + "\n" for row in questions))
            f.write_text("".join(json.dumps(row) + "\n" for row in examples))
            rows = gsm8k_prompts(Tokenizer(), question_file=q, fewshot_file=f, count=5, prompt_tokens=512)
        self.assertEqual([r["request_id"] for r in rows], [f"traj-{i}" for i in range(5)])
        for row in rows:
            text = bytes(row["prompt_ids"]).decode()
            self.assertTrue(text.startswith("<user>")); self.assertTrue(text.endswith("</user><assistant>"))
            self.assertIn(row["question"], text)
            self.assertEqual(len(row["prompt_ids"]), 512)
        for case in ("batch", "preemption_cached", "preemption_recompute", "graph_padding"):
            groups = replay_groups(rows, case, 4)
            self.assertTrue(all(len(group) > 1 for group in groups))
            self.assertEqual({row["request_id"] for group in groups for row in group}, {row["request_id"] for row in rows})
            if case.startswith("preemption_"):
                self.assertEqual([group[-1]["request_id"] for group in groups], [row["request_id"] for row in rows])

    def test_stale_kv_negative_control_detects_wrong_scores_on_cpu(self):
        class CachedRunner:
            def __init__(self):
                self.model = nn.Module()
                self.model.scale = nn.Parameter(torch.tensor(1.))
                self.kv = {}
            def call(self, method, *args):
                if method == "prepare_weight_update":
                    return updates.prepare_weight_update(self.model, args[0])
                if method == "commit_weight_update":
                    return updates.commit_weight_update(args[0])
                seqs, prefill = args
                tokens, scores = [], []
                for seq in seqs:
                    end = seq.num_cached_tokens + seq.num_scheduled_tokens
                    for pos in range(seq.num_cached_tokens, end):
                        self.kv[(seq.block_table[pos // 4], pos % 4)] = seq.token_ids[pos] * self.model.scale.detach()
                    total = sum(self.kv[(seq.block_table[pos // 4], pos % 4)] for pos in range(end))
                    logits = torch.arange(100).float() * total * .0001
                    token = seq.next_forced_token
                    tokens.append(token); scores.append(float(logits.log_softmax(-1)[token]))
                return tokens, scores
        def make_engine():
            engine = cpu_engine(budget=32)
            engine.model_runner = CachedRunner()
            engine.scheduler.record_trace = True
            return engine
        def generate(engine, label):
            row = engine.generate([list(range(1, 13))], SamplingParams(ignore_eos=True),
                request_ids=[label], forced_response_ids=[[21, 22]], use_tqdm=False)[0]
            return {**row, "request_id": "canonical"}
        old = Sequence.block_size
        Sequence.block_size = 4
        try:
            fresh = make_engine()
            fresh.update_weights({"scale": torch.tensor(2.)})
            reference = generate(fresh, "fresh-B")
            for skip in (False, True):
                engine = make_engine()
                generate(engine, "A")
                context = patch.object(engine.scheduler.block_manager, "invalidate", lambda: None)
                if skip:
                    with context:
                        engine.update_weights({"scale": torch.tensor(2.)})
                else:
                    engine.update_weights({"scale": torch.tensor(2.)})
                engine.scheduler.trace.clear()
                output = generate(engine, "B")
                reused = sum(e["tokens"] for e in engine.scheduler.trace if e["event"] == "prefix_reuse")
                if skip:
                    self.assertGreater(reused, 0)
                    with self.assertRaises(RuntimeError):
                        assert_outputs_close([output], [reference], atol=1e-5, rtol=0)
                else:
                    self.assertEqual(reused, 0)
                    assert_outputs_close([output], [reference], atol=1e-5, rtol=0)
        finally:
            Sequence.block_size = old


if __name__ == "__main__":
    unittest.main()
