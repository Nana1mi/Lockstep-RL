import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from lockstep_rl.batch import collate_rollouts
from lockstep_rl.grpo import GRPOConfig, clipped_surrogate, group_advantages
from lockstep_rl.trainer import GRPOTrainer, run_iteration
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.weight_update import prepare_weight_update, commit_weight_update
from tests.test_rollout import cpu_engine


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 7)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embed(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=torch.tanh(hidden.cumsum(1)))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.head = nn.Linear(7, 11, bias=False)

    def get_output_embeddings(self):
        return self.head


def records():
    return [{"request_id": str(i), "group_id": "g", "prompt_ids": [1, 2],
             "response_ids": response, "response_logprobs": [-2.] * len(response),
             "loss_mask": [1] * len(response), "policy_version": 0,
             "sampling_config": {"temperature": 1, "diagnostic_forced": False}}
            for i, response in enumerate(([3], [4, 5, 6], [7, 3], [8, 4, 2, 9]))]


class GRPOTest(unittest.TestCase):
    def test_population_std_and_explicit_interleaved_groups(self):
        adv, stats = group_advantages([1, 3, 2, 3], ["a", "b", "a", "b"], group_size=2)
        torch.testing.assert_close(adv, torch.tensor([-1., 0., 1., 0.]), atol=3e-6, rtol=0)
        self.assertEqual(stats["zero_variance_groups"], 1)
        with self.assertRaises(ValueError):
            group_advantages([1], ["a"], group_size=2)

    def test_clip_signs_and_actual_clip_fraction(self):
        current = torch.log(torch.tensor([[1.5], [1.5], [.5], [.5]])).requires_grad_()
        anchor = torch.zeros_like(current, requires_grad=True)
        advantages = torch.tensor([1., -1., 1., -1.], requires_grad=True)
        loss, stats = clipped_surrogate(current, anchor, advantages, torch.ones_like(current))
        loss.backward()
        self.assertAlmostEqual(loss.item(), .15, places=6)
        self.assertEqual((stats["ratio_outside"], stats["clip_active"]), (4, 2))
        torch.testing.assert_close(current.grad.flatten(), torch.tensor([0., .375, -.125, 0.]))
        self.assertIsNone(anchor.grad)
        self.assertIsNone(advantages.grad)

    def test_padding_and_next_token_shift(self):
        batch = collate_rollouts(records(), pad_token_id=0)
        self.assertEqual(batch.response_mask.sum().item(), 10)
        self.assertEqual(batch.response_mask[0].tolist(), [False, True, False, False, False])
        self.assertEqual(batch.input_ids[0].tolist(), [1, 2, 3, 0, 0, 0])
        current = torch.tensor([[float("nan"), -.2]], requires_grad=True)
        loss, _ = clipped_surrogate(current, torch.zeros_like(current), torch.tensor([1.]), torch.tensor([[0, 1]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(current.grad[0, 0].item(), 0.)

    def test_uneven_microbatch_accumulation_matches_full_token_objective(self):
        torch.manual_seed(11)
        model = ToyLM()
        other = copy.deepcopy(model)
        results = []
        for candidate, size in ((model, 1), (other, 4)):
            trainer = GRPOTrainer(candidate, torch.optim.SGD(candidate.parameters(), lr=.01),
                                  config=GRPOConfig(anchor="recomputed", microbatch_size=size, max_grad_norm=100))
            results.append(trainer.update(records(), [0, 1, 2, 4]))
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
        self.assertAlmostEqual(results[0]["steps"][0]["loss"], results[1]["steps"][0]["loss"], places=6)

    def test_multiple_updates_freeze_anchor_but_recompute_current(self):
        torch.manual_seed(12)
        for anchor in ("rollout", "recomputed"):
            model = ToyLM()
            trainer = GRPOTrainer(model, torch.optim.SGD(model.parameters(), lr=.03),
                                  config=GRPOConfig(anchor=anchor, update_steps=2))
            result = trainer.update(records(), [0, 1, 2, 4])
            self.assertEqual(len(result["steps"]), 2)
            self.assertNotEqual(result["steps"][0]["current_anchor_ratio_mean"],
                                result["steps"][1]["current_anchor_ratio_mean"])
            if anchor == "recomputed":
                self.assertAlmostEqual(result["steps"][0]["current_anchor_ratio_mean"], 1., places=6)
            else:
                self.assertNotEqual(result["steps"][0]["current_anchor_ratio_mean"], 1.)
                self.assertTrue(all(score == -2. for row in result["frozen_anchors"] for score in row))

    def test_zero_signal_skips_optimizer_and_empty_masks_skip(self):
        model = ToyLM()
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = GRPOTrainer(model, optimizer)
        self.assertEqual(trainer.update(records(), [1, 1, 1, 1])["skip_reason"], "zero_advantages")
        self.assertFalse(optimizer.state)
        empty = records()
        for r in empty:
            r["loss_mask"] = [0] * len(r["response_ids"])
        self.assertEqual(trainer.update(empty, [0, 1, 2, 3])["skip_reason"], "empty_mask")
        self.assertFalse(optimizer.state)

    def test_full_cpu_rollout_update_sync_next_rollout_loop(self):
        torch.manual_seed(42)
        model = ToyLM()
        actor = copy.deepcopy(model)
        engine = cpu_engine(blocks=64, budget=16)
        class Runner:
            def call(self, method, *args):
                if method == "prepare_weight_update":
                    return prepare_weight_update(actor, args[0])
                if method == "commit_weight_update":
                    return commit_weight_update(args[0])
                if method == "run":
                    tokens, scores = [], []
                    with torch.no_grad():
                        for seq in args[0]:
                            hidden = actor.model(torch.tensor([seq.token_ids])).last_hidden_state
                            logp = actor.head(hidden[:, -1]).float().log_softmax(-1)[0]
                            token = torch.multinomial(logp.exp(), 1).item()
                            tokens.append(token); scores.append(logp[token].item())
                    return tokens, scores
                raise AssertionError(method)
        engine.model_runner = Runner()
        trainer = GRPOTrainer(model, torch.optim.AdamW(model.parameters(), lr=.003),
                              config=GRPOConfig(anchor="recomputed", update_steps=2))
        old_size = Sequence.block_size
        Sequence.block_size = 4
        try:
            for iteration in range(2):
                batch, result = run_iteration(engine, trainer, [[1, 2]],
                                              lambda r, _: sum(r["response_ids"]),
                                              iteration=iteration, max_tokens=3)
                self.assertTrue(result["updated"])
                self.assertEqual({r["policy_version"] for r in batch}, {iteration})
                self.assertEqual(engine.policy_version, iteration + 1)
                for a, b in zip(actor.parameters(), model.parameters()):
                    torch.testing.assert_close(a, b)
        finally:
            Sequence.block_size = old_size


if __name__ == "__main__":
    unittest.main()
