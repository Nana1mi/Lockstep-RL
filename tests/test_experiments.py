import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from lockstep_rl.experiment_io import exact_chat_prompt, provenance
from lockstep_rl.data import chat_prompts
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import reset_context, set_context
from scripts.diagnose import build_plan, validate_settings
from scripts.weight_sync_experiment import build_plan as sync_plan, assert_outputs_close
from scripts.benchmark_suite import BASELINE, build_plan as benchmark_plan, materialize_baseline


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTest(unittest.TestCase):
    def test_tokenizer_failure_precedes_gpu_runner_and_shared_state_changes(self):
        hf = SimpleNamespace(max_position_embeddings=64, dtype=torch.bfloat16, vocab_size=100)
        block_size = Sequence.block_size
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors").write_bytes(b"")
            with patch("transformers.AutoConfig.from_pretrained", return_value=hf), \
                 patch("transformers.AutoTokenizer.from_pretrained", side_effect=ValueError("invalid tokenizer")), \
                 patch("nanovllm.engine.model_runner.ModelRunner") as runner:
                with self.assertRaisesRegex(ValueError, "tokenizer"):
                    LLMEngine(directory, kvcache_block_size=512)
                runner.assert_not_called()
        self.assertEqual(Sequence.block_size, block_size)
        self.assertFalse(torch.cuda.is_initialized())

    def test_prefill_evidence_distinguishes_request_batch_and_token_count(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.enforce_eager = True
        class Model:
            def __call__(self, ids, positions):
                return ids.float()
            def compute_logits(self, hidden):
                return hidden
        runner.model = Model()
        set_context(True, cu_seqlens_q=torch.tensor([0, 3, 7]))
        try:
            runner.run_model(torch.arange(7), torch.arange(7), True)
            self.assertEqual(runner.last_execution["actual_batch"], 2)
            self.assertEqual(runner.last_execution["input_tokens"], 7)
            runner.run_model(torch.arange(2), torch.arange(2), False)
            self.assertEqual(runner.last_execution["actual_batch"], 2)
            self.assertEqual(runner.last_execution["input_tokens"], 2)
        finally:
            reset_context()
        self.assertFalse(torch.cuda.is_initialized())

    def test_runner_does_not_destroy_a_preexisting_group_when_init_is_rejected(self):
        config = SimpleNamespace(hf_config=SimpleNamespace(dtype=torch.bfloat16), kvcache_block_size=256,
                                 enforce_eager=True, tensor_parallel_size=1, model="unused")
        with patch("nanovllm.engine.model_runner.dist.init_process_group", side_effect=ValueError("existing group")), \
             patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=True), \
             patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy:
            with self.assertRaisesRegex(ValueError, "existing"):
                ModelRunner(config, 0, [])
            destroy.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_baseline_export_uses_exact_fixed_git_source_without_current_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "baseline"
            materialize_baseline(destination)
            self.assertEqual((destination / "BASELINE_COMMIT").read_text().strip(), BASELINE)
            sampler = (destination / "nanovllm/layers/sampler.py").read_text()
            self.assertIn("probs.div_", sampler)
            self.assertNotIn("sample_with_logprobs", sampler)
            self.assertFalse((destination / "lockstep_rl").exists())
            with self.assertRaises(ValueError):
                materialize_baseline(destination)

    def test_force_eager_stance_independently_controls_torch_compile_on_cpu(self):
        calls = []
        def backend(graph, inputs):
            calls.append(True)
            return graph.forward
        compiled = torch.compile(lambda x: x + 1, backend=backend)
        with torch.compiler.set_stance("force_eager"):
            torch.testing.assert_close(compiled(torch.ones(2)), torch.full((2,), 2.))
        self.assertFalse(calls)
        with torch.compiler.set_stance("default"):
            torch.testing.assert_close(compiled(torch.ones(2)), torch.full((2,), 2.))
        self.assertEqual(len(calls), 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_engine_config_and_missing_weights_fail_before_cuda(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            LLMEngine("missing", gpu_memory_utilizaton=.9)
        with patch("torch.distributed.is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "existing process group"):
                LLMEngine("missing")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "safetensors"):
                Config(directory)
            with self.assertRaisesRegex(ValueError, "integer"):
                Config(directory, max_num_seqs=1.5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_vs_current_workloads_and_nested_repeats_match(self):
        settings = json.loads((ROOT / "configs/inference-benchmark.json").read_text())
        jobs = benchmark_plan("config.json", "unused-output", settings)
        self.assertEqual(len(jobs), 96)
        for source in ("baseline", "current"):
            selected = [job for job in jobs if job["source"] == source]
            self.assertEqual(len(selected), 24 if source == "baseline" else 72)
            self.assertEqual({job["process_repeat"] for job in selected}, {0, 1, 2})
            self.assertEqual({job["case"] for job in selected}, set(settings["cases"]))
        self.assertEqual(sum(job["fp32_lm_head"] for job in jobs), 48)
        self.assertEqual(sum(job["cache_fp32_lm_head"] for job in jobs), 24)
        self.assertFalse(torch.cuda.is_initialized())

    def test_sync_plan_locks_repeated_fresh_references_before_hot(self):
        jobs = sync_plan("config.json", "unused-output", 3)
        self.assertEqual(len(jobs), 25)
        for backend in ("eager", "graph"):
            selected = [job for job in jobs if job.get("backend") == backend]
            self.assertEqual([job["mode"] for job in selected[:3]], ["fresh"] * 3)
            self.assertEqual({job["source"] for job in selected if job["mode"] == "hot"}, {"gpu-fp32", "cpu-bf16"})
            self.assertTrue(all("--skip-invalidate" in job["command"] for job in selected if job["mode"] == "negative"))
        row = {"request_id": "target", "prompt_ids": [1], "response_ids": [2], "response_logprobs": [-2.]}
        close = {**row, "response_logprobs": [-2.0001]}
        assert_outputs_close([row], [close], atol=.001, rtol=0)
        with self.assertRaises(RuntimeError):
            assert_outputs_close([row], [close], atol=.00001, rtol=0)

    def test_full_diagnostic_plan_is_cpu_safe_and_covers_required_paths(self):
        path = ROOT / "configs/diagnostics.json"
        settings = json.loads(path.read_text())
        validate_settings(settings)
        jobs = build_plan(path, Path("unused-output"), settings)
        self.assertEqual(len(jobs), 14)
        self.assertEqual(sum(len(job.get("trajectory_ids", [])) for job in jobs), 32 * 14)
        self.assertEqual({job["name"] for job in jobs[:-2]}, {"sample", *settings["cases"]})
        self.assertEqual(jobs[0]["name"], "sample")
        self.assertEqual(jobs[-2]["name"], "hf-bfloat16")
        self.assertEqual(jobs[-1]["name"], "hf-float32")
        self.assertTrue(all("--allow-gpu" in job["command"] for job in jobs))
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_path_settings_fail_before_gpu_work(self):
        original = json.loads((ROOT / "configs/diagnostics.json").read_text())
        for key, value in (("cases", ["eager"]), ("prompt_tokens", 300),
                           ("response_tokens", 1), ("preemption_kv_blocks", 3)):
            settings = copy.deepcopy(original); settings[key] = value
            with self.assertRaises(ValueError):
                validate_settings(settings)
        for key, value in (("max_num_seqs", 3), ("max_num_batched_tokens", 512), ("max_model_len", 512)):
            settings = copy.deepcopy(original); settings["engine"][key] = value
            with self.assertRaises(ValueError):
                validate_settings(settings)

    def test_exact_chat_length_preserves_template_and_thinking_option(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs != {"tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": False}:
                    raise AssertionError(kwargs)
                return [1, 2] + [3] * len(messages[0]["content"].split()) + [4, 5]
        prompt = exact_chat_prompt(Tokenizer(), 512)
        self.assertEqual(len(prompt), 512)
        self.assertEqual(prompt[:2], [1, 2]); self.assertEqual(prompt[-2:], [4, 5])

    def test_training_chat_tokens_opt_out_of_transformers5_dictionary_default(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self_check = {"tokenize": True, "return_dict": False,
                              "add_generation_prompt": True, "enable_thinking": False}
                if kwargs != self_check:
                    return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
                return [1, len(messages[0]["content"]), 3]
        prompts = chat_prompts(Tokenizer(), [{"prompt": "first"}, {"prompt": "next"}])
        self.assertEqual(prompts, [[1, 5, 3], [1, 4, 3]])
        self.assertTrue(all(isinstance(token, int) for row in prompts for token in row))

    def test_missing_or_incomplete_checkpoint_never_silently_uses_random_weights(self):
        model = nn.Linear(2, 2)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_model(model, directory)
            with self.assertRaises(ValueError):
                provenance(directory)
            file = Path(directory) / "model.safetensors"
            save_file({"weight": torch.ones(1, 2), "bias": torch.zeros(2)}, file)
            with self.assertRaises(ValueError):
                load_model(model, directory)
        torch.testing.assert_close(model.weight, before)

    def test_runner_init_failure_restores_defaults_and_distributed_state(self):
        config = SimpleNamespace(hf_config=SimpleNamespace(dtype=torch.bfloat16), kvcache_block_size=256,
                                 enforce_eager=True, tensor_parallel_size=1, model="unused")
        fake_model = nn.Linear(2, 2)
        with patch("nanovllm.engine.model_runner.dist.init_process_group"), \
             patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=True), \
             patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy, \
             patch("nanovllm.engine.model_runner.torch.cuda.set_device"), \
             patch("nanovllm.engine.model_runner.torch.set_default_device") as device, \
             patch("nanovllm.engine.model_runner.torch.set_default_dtype") as dtype, \
             patch("nanovllm.engine.model_runner.Qwen3ForCausalLM", return_value=fake_model), \
             patch("nanovllm.engine.model_runner.load_model", side_effect=ValueError("injected init failure")):
            with self.assertRaisesRegex(ValueError, "injected"):
                ModelRunner(config, 0, [])
            self.assertEqual(device.call_args.args, (torch.device("cpu"),))
            self.assertEqual(dtype.call_args.args, (torch.float32,))
            destroy.assert_called_once()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
