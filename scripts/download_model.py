"""Download a pinned public Qwen3 checkpoint into the project cache."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-download", action="store_true", help="Explicit manual download opt-in")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, default=Path(".models/Qwen3-0.6B"))
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("Model download requires manual --allow-download")
    revision = HfApi().model_info(args.repo, revision=args.revision).sha
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest_path = args.destination / "_lockstep_model.json"
    manifest = {"repo": args.repo, "revision": revision,
                "local_path": str(args.destination.resolve()), "status": "downloading"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    snapshot_download(args.repo, revision=revision, local_dir=args.destination,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    destination = Path("results/stage0/model.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
