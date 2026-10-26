import unittest
import threading
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear
from nanovllm.utils.weight_update import prepare_weight_update, commit_weight_update
from tests.test_rollout import cpu_engine


class TinyPackedModel(nn.Module):
    packed_modules_mapping = {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"),
                              "v_proj": ("qkv_proj", "v"), "gate_proj": ("gate_up_proj", 0),
                              "up_proj": ("gate_up_proj", 1)}

    def __init__(self):
        super().__init__()
        with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
            self.qkv_proj = QKVParallelLinear(3, 2, 2, 1)
            self.gate_up_proj = MergedColumnParallelLinear(3, [5, 5])
        self.embed_tokens = nn.Embedding(7, 3)
        self.lm_head = nn.Linear(3, 7, bias=False)
        self.lm_head.weight.data = self.embed_tokens.weight.data
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


def source_weights():
    return {"q_proj.weight": torch.full((4, 3), 1.), "k_proj.weight": torch.full((2, 3), 2.),
            "v_proj.weight": torch.full((2, 3), 3.), "gate_proj.weight": torch.full((5, 3), 4.),
            "up_proj.weight": torch.full((5, 3), 5.), "embed_tokens.weight": torch.full((7, 3), 6.)}


class CpuWeightRunner:
    def __init__(self, model, fail_copy=False):
        self.model = model
        self.fail_copy = fail_copy

    def call(self, method, value):
        if method == "prepare_weight_update":
            return prepare_weight_update(self.model, value)
        if method == "commit_weight_update":
            if self.fail_copy:
                commit_weight_update(value[:1])
                raise RuntimeError("injected copy failure")
            return commit_weight_update(value)
        raise AssertionError(method)


class WeightUpdateTest(unittest.TestCase):
    def test_partial_parameter_aliases_are_rejected_before_copy(self):
        model = nn.Module()
        model.a = nn.Parameter(torch.zeros(8))
        model.b = nn.Parameter(model.a.data[2:])
        with self.assertRaisesRegex(ValueError, "partial"):
            prepare_weight_update(model, {"a": torch.ones(8), "b": torch.ones(6)})
        self.assertFalse(model.a.count_nonzero())

    def test_admission_waits_for_commit_and_gets_new_policy_version(self):
        engine = cpu_engine()
        entered, release, started, admitted = (threading.Event() for _ in range(4))
        errors = []
        class Runner(CpuWeightRunner):
            def call(self, method, value):
                if method == "prepare_weight_update":
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test release timeout")
                return super().call(method, value)
        engine.model_runner = Runner(TinyPackedModel())
        def update():
            try:
                engine.update_weights(source_weights())
            except Exception as exc:
                errors.append(exc)
        def add():
            from nanovllm import SamplingParams
            started.set()
            try:
                engine.add_request([1, 2], SamplingParams(max_tokens=2), request_id="concurrent")
                admitted.set()
            except Exception as exc:
                errors.append(exc)
        updater, admission = threading.Thread(target=update), threading.Thread(target=add)
        updater.start()
        try:
            self.assertTrue(entered.wait(5))
            admission.start()
            self.assertTrue(started.wait(5))
            self.assertFalse(admitted.wait(.05))
        finally:
            release.set()
            updater.join(5)
            if admission.ident is not None:
                admission.join(5)
        self.assertFalse(errors)
        self.assertTrue(admitted.is_set())
        self.assertEqual(engine.scheduler.waiting[0].policy_version, 1)

    def test_real_upstream_packed_loaders_and_tied_storage(self):
        model = TinyPackedModel()
        pointers = [p.data_ptr() for p in model.parameters()]
        source = source_weights()
        plan = prepare_weight_update(model, source)
        self.assertTrue(all(not p.count_nonzero() for p in model.parameters()))
        commit_weight_update(plan)
        torch.testing.assert_close(model.qkv_proj.weight, torch.cat([source[f"{x}_proj.weight"] for x in "qkv"]))
        torch.testing.assert_close(model.gate_up_proj.weight, torch.cat([source["gate_proj.weight"], source["up_proj.weight"]]))
        torch.testing.assert_close(model.lm_head.weight, source["embed_tokens.weight"])
        self.assertEqual(model.lm_head.weight.data_ptr(), model.embed_tokens.weight.data_ptr())
        self.assertEqual(pointers, [p.data_ptr() for p in model.parameters()])

    def test_preflight_rejects_every_bad_source_without_mutation(self):
        cases = []
        bad = source_weights(); del bad["k_proj.weight"]; cases.append(bad)
        bad = source_weights(); bad["unexpected"] = torch.ones(1); cases.append(bad)
        bad = source_weights(); bad["v_proj.weight"] = torch.ones(3, 3); cases.append(bad)
        bad = source_weights(); bad["q_proj.weight"] = torch.ones(4, 3, dtype=torch.int64); cases.append(bad)
        bad = source_weights(); bad["q_proj.weight"][0, 0] = float("nan"); cases.append(bad)
        bad = source_weights(); bad["lm_head.weight"] = torch.zeros(7, 3); cases.append(bad)
        for source in cases:
            model = TinyPackedModel()
            with self.assertRaises(ValueError):
                prepare_weight_update(model, source)
            self.assertTrue(all(not p.count_nonzero() for p in model.parameters()))

    def test_dtype_conversion_and_overflow_preflight(self):
        model = TinyPackedModel().to(torch.float16)
        # Module.to casts distinct Parameters separately, so explicitly restore
        # the storage tie (the inference model is constructed directly in dtype).
        model.lm_head.weight.data = model.embed_tokens.weight.data
        source = source_weights()
        commit_weight_update(prepare_weight_update(model, source))
        self.assertEqual(model.qkv_proj.weight.dtype, torch.float16)
        source["q_proj.weight"][0, 0] = 1e10
        before = model.qkv_proj.weight.clone()
        with self.assertRaises(ValueError):
            prepare_weight_update(model, source)
        torch.testing.assert_close(model.qkv_proj.weight, before)

    def test_success_commits_version_and_invalidates_logical_cache(self):
        engine = cpu_engine()
        engine.model_runner = CpuWeightRunner(TinyPackedModel())
        manager = engine.scheduler.block_manager
        manager.blocks[0].update(123, [1, 2, 3, 4])
        manager.hash_to_block_id[123] = 0
        self.assertEqual(engine.update_weights(source_weights()), 1)
        self.assertFalse(manager.hash_to_block_id)
        self.assertEqual(engine.state, "ready")
        self.assertEqual(engine.update_weights(source_weights(), policy_version=4), 4)
        with self.assertRaises(ValueError):
            engine.update_weights(source_weights(), policy_version=4)

    def test_preflight_failure_keeps_old_version_and_copy_failure_breaks_engine(self):
        engine = cpu_engine()
        engine.model_runner = CpuWeightRunner(TinyPackedModel())
        with self.assertRaises(ValueError):
            engine.update_weights({})
        self.assertEqual((engine.state, engine.policy_version), ("ready", 0))
        engine.model_runner.fail_copy = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            engine.update_weights(source_weights())
        self.assertEqual((engine.state, engine.policy_version), ("broken", 0))
        for operation in (lambda: engine.step(), lambda: engine.generate([], None),
                          lambda: engine.add_request([1], None),
                          lambda: engine.update_weights(source_weights())):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                operation()

    def test_active_request_and_live_cache_prevent_update(self):
        engine = cpu_engine()
        engine.model_runner = CpuWeightRunner(TinyPackedModel())
        engine.scheduler.waiting.append(object())
        with self.assertRaisesRegex(RuntimeError, "idle"):
            engine.update_weights(source_weights())
        self.assertEqual(engine.policy_version, 0)
        manager = engine.scheduler.block_manager
        manager._allocate_block()
        with self.assertRaisesRegex(RuntimeError, "live"):
            manager.invalidate()


if __name__ == "__main__":
    unittest.main()
