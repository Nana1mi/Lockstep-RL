import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_rental import build_plan, validate_assets
from scripts.prepare_rental_assets import asset_plan
from scripts.verify_gpu_wheel import verify_wheel

ROOT = Path(__file__).resolve().parents[1]


class RentalPlanTest(unittest.TestCase):
    def test_partial_assets_and_leaked_splits_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/Qwen3-0.6B"
            model.mkdir(parents=True)
            (root / "assets-ready.json").write_text(json.dumps({"status": "downloaded_not_gpu_validated"}))
            path = model / "_lockstep_model.json"
            path.write_text(json.dumps({"status": "downloading"}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_assets(root)
            path.write_text(json.dumps({"status": "complete"}))
            (model / "fixture.safetensors").write_bytes(b"not real weights, file presence fixture")
            data = root / "data/gsm8k"
            data.mkdir(parents=True)
            rows_by_split = {}
            for name, count in (("train", 64), ("validation", 32), ("diagnostic", 32), ("fewshot", 32)):
                rows = [{"id": f"{name}-{i}", "prompt": f"{name} fixture {i}", "answer": "1", "solution": "#### 1"} for i in range(count)]
                rows_by_split[name] = rows
                (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            validate_assets(root)
            rows = rows_by_split["validation"]
            rows[0]["prompt"] = rows_by_split["train"][0]["prompt"]
            (data / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_assets(root)

    def test_order_persistent_outputs_and_both_demo_learning_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            persistent = Path(directory) / "not-created"
            plan = build_plan(persistent, "review-01", "/venv/bin/python", "rental fixed mode")
            self.assertFalse(persistent.exists())
        stages = list(dict.fromkeys(job["stage"] for job in plan["jobs"]))
        self.assertEqual(stages, ["environment", "baseline", "diagnostics", "weight_sync", "grpo", "performance"])
        self.assertEqual(plan["hardware"], "NVIDIA A100-SXM4-80GB")
        self.assertTrue(all("--allow-gpu" in job["command"] for job in plan["jobs"]))
        self.assertEqual(len(plan["jobs"]), 13)
        for name, rate in (("train-demo-simple", 1e-6), ("train-demo-simple-lr5e-6", 5e-6)):
            demo = plan["configs"][name]
            self.assertEqual(demo["learning_rate"], rate)
            self.assertGreaterEqual(demo["iterations"], 100)
            self.assertEqual((demo["train_count"], demo["validation_count"]), (512, 128))
            self.assertEqual(demo["grpo"]["microbatch_size"], 16)
            self.assertEqual(demo["grpo"]["gradient_accumulation_steps"], 8)
            self.assertFalse(demo["gradient_checkpointing"])
        low, high = plan["estimate"]["with_30_percent_contingency"]
        self.assertGreater(high, low)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wheel_abi_hash_and_no_download_asset_plan(self):
        config = json.loads((ROOT / "configs/rental.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / config["wheel_filename"]
            fake.write_bytes(b"unit test wheel fixture, not a real binary")
            settings = copy.deepcopy(config)
            settings["wheel_sha256"] = hashlib.sha256(fake.read_bytes()).hexdigest()
            arguments = dict(torch_version="2.8.0+cu128", torch_cuda="12.8", cxx11_abi=True,
                             python_version=(3, 11), machine="x86_64")
            self.assertEqual(verify_wheel(fake, settings, **arguments)["status"], "passed")
            for key, value in (("torch_version", "2.11.0+cu130"), ("cxx11_abi", False), ("python_version", (3, 12))):
                with self.assertRaises(ValueError):
                    verify_wheel(fake, settings, **{**arguments, key: value})
            with self.assertRaises(ValueError):
                verify_wheel(fake, config, **arguments)
        jobs = asset_plan("/persistent/lockstep", "/cpu/python")
        self.assertTrue(all("--allow-download" in job for job in jobs))
        self.assertFalse(any("--allow-gpu" in job for job in jobs))


if __name__ == "__main__":
    unittest.main()
