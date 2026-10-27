import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CpuBoundariesTest(unittest.TestCase):
    def test_public_imports_never_initialize_cuda(self):
        code = """
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA initialization during import')
torch.cuda._lazy_init = forbidden
import nanovllm
from nanovllm.utils.weight_update import prepare_weight_update
from lockstep_rl.logprobs import selected_logprobs
assert not torch.cuda.is_initialized()
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

    def test_gpu_and_download_entries_require_explicit_manual_opt_in(self):
        entries = [(["scripts/attention_smoke.py", "--output", "unused.json"], "--allow-gpu"),
                   (["scripts/setup_managed_python.py", "--persistent-root", "/unused"], "--allow-install"),
                   (["scripts/run_rental.py", "--persistent-root", "/unused"], "--allow-gpu"),
                   (["scripts/prepare_rental_assets.py", "--persistent-root", "/unused"], "--allow-download"),
                   (["scripts/training_memory_probe.py", "--model", "nonexistent", "--output", "unused.json"], "--allow-gpu"),
                   (["bench.py"], "--allow-gpu"), (["example.py"], "--allow-gpu"),
                   (["scripts/train_grpo.py"], "--allow-gpu"),
                   (["scripts/evaluate.py", "--model", "nonexistent"], "--allow-gpu"),
                   (["scripts/score_hf.py", "--model", "nonexistent", "--trajectory", "unused.json", "--output", "unused.json"], "--allow-gpu"),
                   (["scripts/diagnose.py"], "--allow-gpu"),
                   (["scripts/weight_sync_experiment.py"], "--allow-gpu"),
                   (["scripts/benchmark_suite.py"], "--allow-gpu"),
                   (["scripts/sampler_benchmark.py"], "--allow-gpu"),
                   (["scripts/inference_benchmark_worker.py", "--config", "configs/inference-benchmark.json", "--case", "short-single", "--source", "current", "--source-path", ".", "--backend", "eager", "--output", "unused.json"], "--allow-gpu"),
                   (["scripts/weight_sync_worker.py", "--config", "configs/weight-sync.json", "--mode", "hot", "--backend", "eager", "--model-b", "unused", "--trajectory", "unused.json", "--output-dir", "unused-output"], "--allow-gpu"),
                   (["scripts/nano_path_worker.py", "--config", "configs/diagnostics.json", "--case", "sample", "--output-dir", "unused-output"], "--allow-gpu"),
                   (["scripts/download_model.py"], "--allow-download"),
                   (["scripts/prepare_gsm8k.py"], "--allow-download")]
        for argv, expected in entries:
            result = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(expected, result.stderr)
        self.assertFalse((ROOT / "unused.json").exists())

    def test_training_dry_run_without_gpu_or_model(self):
        result = subprocess.run([sys.executable, "scripts/train_grpo.py", "--dry-run"],
                                cwd=ROOT, text=True, capture_output=True,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU_PENDING", result.stdout)


if __name__ == "__main__":
    unittest.main()
