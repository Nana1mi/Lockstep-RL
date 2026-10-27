"""CPU acceptance for L40S placeholders, source integrity and offline setup."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lockstep_rl.rental import data_mount, file_hashes, sha256, validate_gpu, validate_preflight, verify_hashes
from scripts.prepare_rental_assets import asset_plan
from scripts.setup_managed_python import prepare_managed_python

ROOT = Path(__file__).resolve().parents[1]


class L40SRentalTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT / "configs/rental.json").read_text(encoding="utf-8"))
        self.gpu = {"name": "NVIDIA L40S", "uuid": "synthetic-fixture", "driver": "570.26", "memory": "46068 MiB"}

    def test_target_placeholders_wheel_pins_and_memory_budget(self):
        self.assertEqual(self.settings["gpu_name"], "L40S")
        self.assertIsNone(self.settings["expected_gpu_name"])
        self.assertTrue(all(value is None for value in self.settings["preflight_pending"]["network"].values()))
        self.assertEqual(self.settings["architecture"], "Ada sm_89")
        self.assertEqual(self.settings["wheel_sha256"], "92aabf18ecfcf5d620ac7fe97fafe88f83e714b7a790e4bf219dff4a404cfb38")
        self.assertIn("v2.8.3.post2/", self.settings["wheel_url"])
        for path in (ROOT / "configs").glob("train-*.json"):
            config = json.loads(path.read_text())
            self.assertLessEqual(config["engine"]["gpu_memory_utilization"], .4)
            if "demo" in path.name:
                self.assertEqual(config["engine"]["gpu_memory_utilization"], .4)
                self.assertEqual(config["grpo"]["microbatch_size"], 16)
                self.assertFalse(config["gradient_checkpointing"])

    def test_exact_name_and_driver_gate_reject_guesses_and_old_drivers(self):
        validate_gpu(self.gpu, "NVIDIA L40S", self.settings)
        for name in (None, "L40S", "NVIDIA A100-SXM4-80GB", "USER_MUST_CONFIRM"):
            with self.assertRaises(ValueError):
                validate_gpu(self.gpu, name, self.settings)
        for driver in ("569.99", "535.230", "unknown", ""):
            with self.assertRaises(ValueError):
                validate_gpu({**self.gpu, "driver": driver}, "NVIDIA L40S", self.settings)

    def test_region_identity_and_mount_must_match_confirmed_record(self):
        mount = {"persistent_root": "/data/lockstep", "data": {"target": "/data", "maj:min": "8:16"}}
        record = {"status": "confirmed", "region": "安徽区", "expected_gpu_name": self.gpu["name"], "gpu": self.gpu, "mount": mount}
        reader = lambda _: mount
        self.assertEqual(validate_preflight(record, "/data/lockstep", self.gpu, self.settings, mount_reader=reader), record)
        for changed in ({**record, "region": "other"}, {**record, "status": "pending"},
                        {**record, "gpu": {**self.gpu, "uuid": "another-card"}}, {**record, "mount": {}}):
            with self.assertRaises(ValueError):
                validate_preflight(changed, "/data/lockstep", self.gpu, self.settings, mount_reader=reader)

    def test_findmnt_rejects_system_disk_and_same_filesystem_bind_mount(self):
        system = {"target": "/", "source": "/dev/root", "fstype": "ext4", "maj:min": "8:1"}
        data = {"target": "/data", "source": "/dev/data", "fstype": "ext4", "maj:min": "8:16"}
        for found, accepted in ((data, True), (system, False), ({**data, "maj:min": "8:1"}, False)):
            outputs = [SimpleNamespace(stdout=json.dumps({"filesystems": [value]})) for value in (found, system)]
            with patch("lockstep_rl.rental.subprocess.run", side_effect=outputs):
                if accepted:
                    self.assertEqual(data_mount("/data/project")["data"], data)
                else:
                    with self.assertRaises(ValueError):
                        data_mount("/data/project")

    def test_hash_checks_detect_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.jsonl"; path.write_bytes(b"tiny CPU fixture")
            hashes = file_hashes(root, [path])
            verify_hashes(root, hashes)
            path.write_bytes(b"modified fixture")
            with self.assertRaises(ValueError):
                verify_hashes(root, hashes)
            with self.assertRaises(ValueError):
                verify_hashes(root, {"../escape": next(iter(hashes.values()))})

    def test_mirror_and_revision_propagate_to_both_assets_without_execution(self):
        jobs = asset_plan("/data/project", "/cpu/python", hf_endpoint="https://hf-mirror.com",
                          model_revision="model-commit", dataset_revision="dataset-commit")
        for command, revision in zip(jobs, ("model-commit", "dataset-commit")):
            self.assertEqual(command[command.index("--hf-endpoint") + 1], "https://hf-mirror.com")
            self.assertEqual(command[command.index("--revision") + 1], revision)
            self.assertIn("--allow-download", command)

    def test_model_mirror_manifest_records_actual_resolved_revision_and_hashes(self):
        from scripts import download_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "model"
            def snapshot(*args, **kwargs):
                model.joinpath("model.safetensors").write_bytes(b"small fake weights for CPU test")
                model.joinpath("config.json").write_text("{}")
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["download_model", "--allow-download", "--hf-endpoint", "https://hf-mirror.com", "--destination", str(model)]), \
                     patch.object(download_model, "HfApi") as api, patch.object(download_model, "snapshot_download", side_effect=snapshot) as fetch, patch("builtins.print"):
                    api.return_value.model_info.return_value.sha = "a" * 40
                    download_model.main()
                api.assert_called_once_with(endpoint="https://hf-mirror.com")
                self.assertEqual(fetch.call_args.kwargs["revision"], "a" * 40)
                manifest = json.loads((model / "_lockstep_model.json").read_text())
                self.assertEqual(manifest["source"]["endpoint"], "https://hf-mirror.com")
                self.assertEqual(manifest["status"], "complete")
                verify_hashes(model, manifest["files"])
            finally:
                os.chdir(cwd)

    def test_dataset_mirror_resolves_same_endpoint_and_hashes_subsets(self):
        from scripts import prepare_gsm8k

        dataset = {split: [{"question": f"{split} question {i}", "answer": "#### 1"} for i in range(4)] for split in ("train", "test")}
        fake_hub = SimpleNamespace(HfApi=Mock())
        fake_hub.HfApi.return_value.dataset_info.return_value.sha = "b" * 40
        seen = []
        def load(*args, **kwargs):
            seen.append((os.environ["HF_ENDPOINT"], kwargs["revision"]))
            return dataset
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            args = ["dataset", "--allow-download", "--hf-endpoint", "https://hf-mirror.com", "--output-dir", directory]
            for split in ("train", "validation", "diagnostic", "fewshot"):
                args += ["--" + split + "-count", "1"]
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load), "huggingface_hub": fake_hub}):
                prepare_gsm8k.main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(seen, [("https://hf-mirror.com", "b" * 40)])
            self.assertEqual(manifest["source"]["endpoint"], "https://hf-mirror.com")
            self.assertEqual(len(manifest["files"]), 4)
            verify_hashes(directory, manifest["files"])

    def test_external_conda_prefix_is_persistent_and_host_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exe = root / "py311/bin/python"
            exe.parent.mkdir(parents=True); exe.touch()
            probe = {"version": [3, 11], "base_prefix": str(root / "py311"), "executable": str(exe)}
            with patch("scripts.setup_managed_python.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(probe))) as process:
                report = prepare_managed_python(root, python=exe, provider="conda")
            self.assertEqual(report["provider"], "conda")
            self.assertEqual(process.call_count, 1)
            self.assertEqual(report["commands"], [])
        with tempfile.TemporaryDirectory() as directory:
            probe["base_prefix"] = "/usr"
            with patch("scripts.setup_managed_python.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(probe))):
                with self.assertRaises(ValueError):
                    prepare_managed_python(directory, python=Path(directory) / "py311/bin/python", provider="conda")

    def test_uploaded_python_requires_archive_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload.tar.gz"; archive.write_bytes(b"archive fixture")
            with self.assertRaises(ValueError):
                prepare_managed_python(root, python=root / "python/bin/python", provider="uploaded", source_archive=archive, source_sha256="0" * 64)

    def test_uploaded_python_records_verified_archive_without_uv_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload.tar.gz"; archive.write_bytes(b"archive fixture")
            exe = root / "python-upload/python/bin/python3.11"; exe.parent.mkdir(parents=True); exe.touch()
            probe = {"version": [3, 11], "base_prefix": str(exe.parent.parent), "executable": str(exe)}
            with patch("scripts.setup_managed_python.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(probe))) as process:
                report = prepare_managed_python(root, python=exe, provider="uploaded", source_archive=archive, source_sha256=sha256(archive))
            self.assertEqual(process.call_count, 1)
            self.assertEqual(report["source_archive"]["sha256"], sha256(archive))
            self.assertEqual(report["provider"], "uploaded")

    def test_uploaded_official_wheel_is_verified_without_github_access(self):
        from scripts import prepare_rental_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); workspace = root / "workspace"; (workspace / "configs").mkdir(parents=True)
            wheel = root / self.settings["wheel_filename"]; wheel.write_bytes(b"fake CPU wheel fixture")
            settings = {**self.settings, "wheel_sha256": sha256(wheel)}
            (workspace / "configs/rental.json").write_text(json.dumps(settings))
            persistent = root / "data"
            args = ["assets", "--allow-download", "--persistent-root", str(persistent), "--wheel", str(wheel), "--hf-endpoint", "https://hf-mirror.com"]
            with patch.object(prepare_rental_assets, "ROOT", workspace), patch.object(sys, "argv", args), \
                 patch.object(prepare_rental_assets.urllib.request, "urlopen", side_effect=AssertionError("must stay offline for wheel")), \
                 patch.object(prepare_rental_assets.subprocess, "run") as process:
                prepare_rental_assets.main()
            self.assertEqual(process.call_count, 2)
            self.assertEqual(sha256(persistent / "wheels" / wheel.name), sha256(wheel))


if __name__ == "__main__":
    unittest.main()
