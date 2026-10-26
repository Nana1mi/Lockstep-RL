import unittest

import torch
from torch import nn

from lockstep_rl.logprobs import selected_logprobs
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import reset_context
from unittest.mock import patch
from nanovllm.utils.weight_update import prepare_weight_update, commit_weight_update


class SelectedLogprobsTest(unittest.TestCase):
    def test_fp32_head_disables_autocast_and_observes_hot_weights(self):
        torch.manual_seed(41)
        with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
            head = ParallelLMHead(29, 11, fp32_lm_head=True).to(torch.bfloat16)
        with torch.no_grad():
            head.weight.normal_()
        hidden = torch.randn(7, 11, dtype=torch.bfloat16)
        targets = torch.arange(7)
        reset_context()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = head(hidden)
            selected = selected_logprobs(hidden, head, targets, fp32_lm_head=True, chunk_size=3)
        reference = torch.nn.functional.linear(hidden.float(), head.weight.float())
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        torch.testing.assert_close(selected, reference.log_softmax(-1).gather(-1, targets[:, None]).flatten(), atol=2e-6, rtol=2e-6)
        pointer = head.weight.data_ptr()
        replacement = torch.randn_like(head.weight)
        commit_weight_update(prepare_weight_update(head, {"weight": replacement}))
        self.assertEqual(head.weight.data_ptr(), pointer)
        torch.testing.assert_close(head(hidden), torch.nn.functional.linear(hidden.float(), replacement.float()), atol=0, rtol=0)
        self.assertFalse(torch.equal(actual, head(hidden)))
        head.weight.requires_grad_(True)
        selected_logprobs(hidden, head, targets, fp32_lm_head=True).sum().backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_values_and_gradients_match_full_log_softmax(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 7, 11, requires_grad=True)
        head = nn.Linear(11, 29, bias=True)
        targets = torch.randint(0, 29, (2, 7))
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0, 0]])
        reference = head(hidden).float().log_softmax(-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        reference_grads = torch.autograd.grad(
            (reference * mask).sum(), (hidden, head.weight, head.bias))
        # Uneven final chunk and a batch dimension that must flatten correctly.
        actual = selected_logprobs(hidden, head, targets, chunk_size=5)
        actual_grads = torch.autograd.grad(
            (actual * mask).sum(), (hidden, head.weight, head.bias))
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        for actual_grad, expected_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)

    def test_frozen_hidden_still_trains_head_and_no_grad_scores_match(self):
        hidden = torch.randn(1, 4, 8)
        head = nn.Linear(8, 19, bias=False)
        targets = torch.tensor([[0, 2, 3, 18]])
        actual = selected_logprobs(hidden, head, targets, chunk_size=3)
        actual.sum().backward()
        self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            frozen = selected_logprobs(hidden, head, targets, chunk_size=3)
        torch.testing.assert_close(actual.detach(), frozen)

    def test_extreme_logits_are_finite(self):
        head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1000., 1000.], [-1000., -1000.], [0., 0.]]))
        actual = selected_logprobs(torch.ones(1, 2), head, torch.tensor([1]))
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual, torch.tensor([-4000.]))

    def test_checkpoint_does_not_save_vocabulary_activations(self):
        hidden = torch.randn(2, 17, 11, requires_grad=True)
        head = nn.Linear(11, 97, bias=False)
        targets = torch.randint(0, 97, (2, 17))
        saved_shapes = []
        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            scores = selected_logprobs(hidden, head, targets, chunk_size=5)
        self.assertFalse(any(97 in shape for shape in saved_shapes), saved_shapes)
        scores.sum().backward()

    def test_bad_inputs(self):
        head = nn.Linear(3, 5)
        with self.assertRaises(ValueError):
            selected_logprobs(torch.randn(2, 3), head, torch.ones(1, dtype=torch.long))
        with self.assertRaises(ValueError):
            selected_logprobs(torch.randn(2, 3), head, torch.ones(2, dtype=torch.long), 0)


if __name__ == "__main__":
    unittest.main()
