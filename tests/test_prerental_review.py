"""Review regressions: EOS populations, cache commit, independent job recovery."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from lockstep_rl.diagnostics import compare_trajectories, eos_boundary
from lockstep_rl.diagnostic_prompts import gsm8k_prompts
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.weight_update import prepare_weight_update, commit_weight_update
from scripts.run_rental import build_plan, execute_jobs, validate_assets
from scripts.setup_managed_python import managed_plan, prepare_managed_python
from scripts.weight_sync_experiment import assert_outputs_close
from tests.test_rollout import cpu_engine
from tests.test_weight_update import CpuWeightRunner
from tests.test_grpo import ToyLM, records
from lockstep_rl.grpo import GRPOConfig
from lockstep_rl.trainer import GRPOTrainer


class PrerentalReviewTest(unittest.TestCase):
    def test_model_only_inputs_do_not_require_dataset_or_global_ready_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/Qwen3-0.6B"
            model.mkdir(parents=True)
            (model / "_lockstep_model.json").write_text(json.dumps({"status": "complete"}))
            (model / "fixture.safetensors").write_bytes(b"file presence fixture only")
            validate_assets(root, splits=[])
            with self.assertRaises(FileNotFoundError):
                validate_assets(root, splits=["diagnostic", "fewshot"])

    def test_demo_128_sequences_microbatch16_matches_full_token_normalization(self):
        torch.manual_seed(15)
        source = ToyLM()
        rows = []
        template = records()
        for index in range(128):
            row = copy.deepcopy(template[index % 4])
            row.update(request_id=str(index), group_id=f"group-{index // 8}")
            rows.append(row)
        rewards = [float(index % 8) for index in range(128)]
        models = [copy.deepcopy(source) for _ in range(2)]
        for model, microbatch, accumulation in zip(models, (16, 128), (8, 1)):
            trainer = GRPOTrainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                config=GRPOConfig(group_size=8, anchor="recomputed", microbatch_size=microbatch,
                                  gradient_accumulation_steps=accumulation))
            self.assertTrue(trainer.update(rows, rewards)["updated"])
        for small, full in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(small, full, atol=2e-6, rtol=2e-6)

    def test_eos_inclusive_population_excludes_extreme_tail_and_exports_it(self):
        rows = [{"request_id": "traj-0", "prompt_ids": [1], "response_ids": [3, 99, 4, 98],
                 "response_logprobs": [-2.] * 4, "sampling_config": {"eos_token_ids": [98, 99]}},
                {"request_id": "traj-1", "prompt_ids": [1], "response_ids": [4, 5],
                 "response_logprobs": [-2., -2.], "sampling_config": {"eos_token_ids": [99]}}]
        other = copy.deepcopy(rows)
        other[0]["response_logprobs"] = [-1.9, -1.8, -102., -52.]
        stats = compare_trajectories(rows, other, threshold=.3)
        self.assertEqual(stats["token_count"], 4)
        self.assertAlmostEqual(stats["absolute_logprob_error"]["max"], .2)
        self.assertEqual(stats["trajectories_above_threshold"], 0)
        self.assertEqual(stats["post_eos"]["token_count"], 2)
        self.assertEqual(stats["post_eos"]["absolute_logprob_error"]["max"], 100.)
        self.assertEqual(stats["eos_boundaries"]["traj-0"]["first_eos_index"], 1)
        self.assertEqual([row["response_index"] for row in stats["post_eos"]["rows"]], [2, 3])
        self.assertEqual(len(stats["all_rows"]), 6)
        same_primary = copy.deepcopy(other)
        same_primary[0]["response_logprobs"][:2] = [-2., -2.]
        assert_outputs_close(rows, same_primary, atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            assert_outputs_close(rows, other, atol=.01, rtol=0)
        first = {**rows[0], "response_ids": [99, 3, 4, 98]}
        self.assertEqual(eos_boundary(first)["primary_response_tokens"], 1)
        no_tail = compare_trajectories([rows[1]], [rows[1]])
        self.assertEqual(no_tail["post_eos"]["token_count"], 0)
        self.assertIsNone(no_tail["post_eos"]["absolute_logprob_error"])

    def make_cached_model(self):
        model = nn.Module()
        model.embed = nn.Embedding(7, 3).to(torch.bfloat16)
        with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
            model.head = ParallelLMHead(7, 3, fp32_lm_head=True, cache_fp32_lm_head=True).to(torch.bfloat16)
        model.head.weight.data = model.embed.weight.data
        return model

    def test_cached_head_initial_load_and_tied_commit_refresh_without_forward_conversion(self):
        torch.manual_seed(7)
        model = self.make_cached_model()
        reset_context()
        hidden = torch.randn(2, 3, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            model.head(hidden)
        source = torch.randn(7, 3, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"embed.weight": source}, Path(directory) / "model.safetensors")
            load_model(model, directory)
        self.assertEqual(model.head._fp32_weight.dtype, torch.float32)
        self.assertFalse(model.head._fp32_weight.requires_grad)
        self.assertNotIn("head._fp32_weight", model.state_dict())
        pointer = model.head._fp32_weight.data_ptr()
        with patch.object(model.head, "refresh_fp32_weight", wraps=model.head.refresh_fp32_weight) as refresh:
            before = model.head(hidden)
            torch.testing.assert_close(before, nn.functional.linear(hidden.float(), source.float()), atol=0, rtol=0)
            model.head(hidden)
            refresh.assert_not_called()
            replacement = source + 1
            plan = prepare_weight_update(model, {"embed.weight": replacement})
            torch.testing.assert_close(model.head._fp32_weight, source.float(), atol=0, rtol=0)
            commit_weight_update(plan)
            self.assertEqual(refresh.call_count, 1)
            after = model.head(hidden)
        self.assertEqual(model.head._fp32_weight.data_ptr(), pointer)
        torch.testing.assert_close(after, nn.functional.linear(hidden.float(), replacement.float()), atol=0, rtol=0)
        model.head.cache_fp32_lm_head = False
        torch.testing.assert_close(after, model.head(hidden), atol=0, rtol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cache_refresh_failure_breaks_engine_and_sync_timer_excludes_measurement(self):
        engine = cpu_engine()
        model = self.make_cached_model()
        engine.model_runner = CpuWeightRunner(model)
        update = engine.update_weights({"embed.weight": torch.ones(7, 3)})
        times = update["timings"]
        self.assertEqual(times["sync_seconds"], sum(times[k] for k in ("preflight_seconds", "commit_seconds", "invalidate_seconds")))
        with patch.object(model.head, "refresh_fp32_weight", side_effect=RuntimeError("refresh failed")):
            with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                engine.update_weights({"embed.weight": torch.zeros(7, 3)})
        self.assertEqual((engine.state, engine.policy_version), ("broken", 1))
        with self.assertRaisesRegex(RuntimeError, "broken"):
            engine.step()

    def test_rental_soft_failures_continue_and_hard_failures_stop(self):
        inventory = {"name": "fixture", "uuid": "fixture"}
        for failed_name in ("diagnostics", "environment", "attention"):
            with tempfile.TemporaryDirectory() as directory:
                plan = build_plan(directory, "review", "/not/executed/python", "fixture")
                Path(plan["output"]).mkdir(parents=True)
                def run(command, **kwargs):
                    if "scripts/diagnose.py" in command and failed_name == "diagnostics":
                        raise OSError("synthetic child startup failure")
                    script = "scripts/environment_probe.py" if failed_name == "environment" else "scripts/attention_smoke.py"
                    return SimpleNamespace(returncode=7 if failed_name != "diagnostics" and script in command else 0)
                with patch("scripts.run_rental.subprocess.run", side_effect=run) as process:
                    report = execute_jobs(plan, inventory, {}, "0", validate_inputs=False, inventory_reader=lambda _: inventory)
                self.assertEqual(report["failed_jobs"], [failed_name])
                if failed_name == "diagnostics":
                    self.assertEqual(process.call_count, 13)
                    self.assertEqual(report["status"], "completed_with_failures")
                    self.assertEqual(report["completed_jobs"][-1]["name"], "sampler")
                    self.assertTrue((Path(plan["output"]) / "diagnostics.console.log").is_file())
                else:
                    self.assertEqual(report["status"], "blocked_by_environment")
                    self.assertEqual(process.call_count, 1 if failed_name == "environment" else 2)
                    self.assertGreater(len(report["skipped_jobs"]), 0)
                self.assertEqual(json.loads((Path(plan["output"]) / "manifest.json").read_text())["status"], report["status"])

    def test_stage_selection_always_includes_environment_and_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            plan = build_plan(root, "retry", "/fake/python", "fixture", ["sync", "performance"])
            self.assertEqual(plan["selected_stages"], ["environment", "weight_sync", "performance"])
            self.assertEqual([job["name"] for job in plan["jobs"]], ["environment", "attention", "weight-sync", "performance", "sampler"])
            self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                build_plan(root, "retry", "/fake/python", "fixture", ["unknown"])

    def test_managed_python_is_persistent_and_rejects_host_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "python/cpython-3.11/bin/python3.11"
            executable.parent.mkdir(parents=True)
            executable.touch()
            probe = {"version": [3, 11], "base_prefix": str(executable.parent.parent), "executable": str(executable)}
            outputs = [SimpleNamespace(), SimpleNamespace(stdout=str(executable)), SimpleNamespace(stdout=json.dumps(probe))]
            with patch("scripts.setup_managed_python.subprocess.run", side_effect=outputs) as process:
                report = prepare_managed_python(root, "/fixture/uv")
            self.assertEqual(report["status"], "managed_python_ready")
            self.assertEqual(process.call_args_list[0].kwargs["env"]["UV_PYTHON_INSTALL_DIR"], str(root / "python"))
            self.assertEqual((root / "python-executable.txt").read_text().strip(), str(executable))
            with self.assertRaises(ValueError):
                prepare_managed_python(root)
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.setup_managed_python.subprocess.run", side_effect=[SimpleNamespace(), SimpleNamespace(stdout="/usr/bin/python3")]):
                with self.assertRaises(ValueError):
                    prepare_managed_python(directory)
            self.assertFalse((Path(directory) / "python-executable.txt").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "unused"
            self.assertEqual(managed_plan(root)["status"], "INSTALL_PENDING")
            self.assertFalse(root.exists())

    def test_seeded_fewshot_rotation_is_diverse_and_reproducible(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "<user>" + messages[0]["content"] + "</user><assistant>"
            def encode(self, text, **kwargs):
                return list(text.encode())
        with tempfile.TemporaryDirectory() as directory:
            q, f = Path(directory) / "q.jsonl", Path(directory) / "f.jsonl"
            questions = [{"id": str(i), "prompt": f"Target {i}", "answer": "1"} for i in range(4)]
            examples = [{"id": f"example-{i}", "prompt": f"Example {i}", "solution": f"Solved exercise {i}. " * 40, "answer": "1"} for i in range(4)]
            q.write_text("".join(json.dumps(row) + "\n" for row in questions))
            f.write_text("".join(json.dumps(row) + "\n" for row in examples))
            args = dict(question_file=q, fewshot_file=f, count=4, prompt_tokens=256, seed=7)
            rows = gsm8k_prompts(Tokenizer(), **args)
            self.assertEqual(rows, gsm8k_prompts(Tokenizer(), **args))
            self.assertEqual(len({row["fewshot_ids"][-1] for row in rows}), 4)
            self.assertEqual(len({tuple(row["prompt_ids"][:200]) for row in rows}), 4)


if __name__ == "__main__":
    unittest.main()
