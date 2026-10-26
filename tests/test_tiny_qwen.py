import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from lockstep_rl.grpo import GRPOConfig
from lockstep_rl.trainer import GRPOTrainer
from lockstep_rl.batch import collate_rollouts
from lockstep_rl.scoring import load_effective_hf, score_records
from nanovllm.models.qwen3 import Qwen3ForCausalLM as NanoQwen3
from nanovllm.utils.loader import load_model
from nanovllm.utils.weight_update import commit_weight_update, prepare_weight_update
from tests.test_grpo import records


def config():
    return Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=8, max_position_embeddings=32, attention_dropout=0,
                       tie_word_embeddings=True)


class TinyQwenTest(unittest.TestCase):
    def test_teacher_forcing_matches_full_logits_at_response_positions(self):
        torch.manual_seed(8)
        model = Qwen3ForCausalLM(config()).eval()
        rows = records()
        actual = score_records(model, rows, chunk_size=3)
        batch = collate_rollouts(rows, pad_token_id=0)
        with torch.no_grad():
            logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           use_cache=False).logits[:, :-1].float()
            full = logits.log_softmax(-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
        for i, row in enumerate(actual):
            torch.testing.assert_close(torch.tensor(row["response_logprobs"]), full[i][batch.response_mask[i]],
                                       atol=1e-6, rtol=1e-6)
            self.assertTrue(row["diagnostic_scoring"])
        self.assertEqual(rows[0]["response_logprobs"], [-2.])
        trainer = GRPOTrainer(model, torch.optim.SGD(model.parameters(), lr=.01))
        with self.assertRaises(ValueError):
            trainer.update(actual, [0, 1, 2, 3])

    def test_fp32_reference_uses_promoted_bf16_effective_weights(self):
        torch.manual_seed(9)
        original = Qwen3ForCausalLM(config())
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            promoted = load_effective_hf(directory, dtype=torch.float32, device="cpu")
        for left, right in zip(original.parameters(), promoted.parameters()):
            self.assertEqual(right.dtype, torch.float32)
            torch.testing.assert_close(right, left.to(torch.bfloat16).float(), atol=0, rtol=0)

    def test_cpu_hf_qwen_checkpointing_and_two_updates(self):
        torch.manual_seed(5)
        model = Qwen3ForCausalLM(config())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layer = model.model.layers[0]
        original_checkpoint = layer._gradient_checkpointing_func
        calls = []
        def counted(*args, **kwargs):
            calls.append(True)
            return original_checkpoint(*args, **kwargs)
        layer._gradient_checkpointing_func = counted
        trainer = GRPOTrainer(model, torch.optim.AdamW(model.parameters(), lr=.0001),
                              config=GRPOConfig(anchor="recomputed", update_steps=2))
        result = trainer.update(records(), [0, 1, 2, 4])
        self.assertEqual(len(result["steps"]), 2)
        self.assertTrue(model.training)
        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(result["steps"][0]["current_anchor_ratio_mean"], 1., places=6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_hf_keys_match_nano_packed_and_tied_weights(self):
        torch.manual_seed(6)
        hf = Qwen3ForCausalLM(config())
        with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
            hot = NanoQwen3(config())
            fresh = NanoQwen3(config())
        pointers = [p.data_ptr() for p in hot.parameters()]
        commit_weight_update(prepare_weight_update(hot, hf.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            hf.save_pretrained(directory)
            load_model(fresh, directory)
        for (name, parameter), (other_name, other) in zip(hot.named_parameters(), fresh.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
        self.assertEqual(hot.model.embed_tokens.weight.data_ptr(), hot.lm_head.weight.data_ptr())
        self.assertEqual(pointers, [p.data_ptr() for p in hot.parameters()])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
