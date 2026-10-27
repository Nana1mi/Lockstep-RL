"""Manual CPU/network-only asset preparation on a persistent disk, before rental."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lockstep_rl.rental import sha256


def asset_plan(root, python, *, hf_endpoint=None, model_revision="main", dataset_revision="main"):
    root = Path(root)
    jobs = [
        [python, "scripts/download_model.py", "--allow-download", "--destination", str(root / "models/Qwen3-0.6B")],
        [python, "scripts/prepare_gsm8k.py", "--allow-download", "--train-count", "64", "--validation-count", "32",
         "--diagnostic-count", "32", "--fewshot-count", "32", "--output-dir", str(root / "data/gsm8k")],
    ]
    for command, revision in zip(jobs, (model_revision, dataset_revision)):
        command += ["--revision", revision, "--hf-endpoint", hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--wheel", type=Path, help="Official wheel uploaded to disk; verify pinned filename and SHA256, no wheel network download")
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/rental.json").read_text())
    root = args.persistent_root.resolve()
    jobs = asset_plan(root, sys.executable, hf_endpoint=args.hf_endpoint, model_revision=args.model_revision, dataset_revision=args.dataset_revision)
    if args.dry_run:
        print(json.dumps({"status": "DOWNLOAD_PENDING", "commands": jobs, "wheel": str(args.wheel) if args.wheel else settings["wheel_url"],
                          "wheel_sha256": settings["wheel_sha256"], "gpu_execution": False}, indent=2))
        return
    if not args.allow_download:
        parser.error("model/dataset/wheel downloads require manual --allow-download")
    root.mkdir(parents=True, exist_ok=True)
    # The upstream wheel is downloaded without executing any setup/build hook.
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    wheel = wheels / settings["wheel_filename"]
    temporary = wheels / (settings["wheel_filename"] + ".part")
    if args.wheel:
        if args.wheel.name != settings["wheel_filename"] or sha256(args.wheel) != settings["wheel_sha256"]:
            raise ValueError("uploaded official wheel filename/SHA256 mismatch")
        if args.wheel.resolve() != wheel.resolve():
            shutil.copyfile(args.wheel, temporary)
            temporary.replace(wheel)
    if not wheel.exists():
        with urllib.request.urlopen(settings["wheel_url"], timeout=60) as incoming, temporary.open("wb") as outgoing:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
        if sha256(temporary) != settings["wheel_sha256"]:
            raise ValueError("official wheel SHA256 mismatch; partial download retained for investigation")
        temporary.replace(wheel)
    if sha256(wheel) != settings["wheel_sha256"]:
        raise ValueError("persistent wheel SHA256 mismatch")
    for command in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
    (root / "assets-ready.json").write_text(json.dumps({"status": "downloaded_not_gpu_validated",
        "model_manifest": str(root / "models/Qwen3-0.6B/_lockstep_model.json"),
        "dataset_manifest": str(root / "data/gsm8k/manifest.json"),
        "wheel": str(wheel), "wheel_sha256": settings["wheel_sha256"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
