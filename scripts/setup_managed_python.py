"""Manual uv-managed Python on persistent storage; no GPU or environment reuse."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lockstep_rl.rental import sha256


def managed_plan(root, uv="uv"):
    root = Path(root).resolve()
    return {"status": "INSTALL_PENDING", "persistent_root": str(root),
            "environment": {"UV_PYTHON_INSTALL_DIR": str(root / "python"), "UV_CACHE_DIR": str(root / "uv-cache")},
            "commands": [[uv, "python", "install", "--no-bin", "3.11"],
                         [uv, "python", "find", "--managed-python", "3.11"]],
            "note": "Preserve the same absolute mount path/OS/architecture. Recreate venv if those change."}


def prepare_managed_python(root, uv="uv", *, python=None, provider="uv", source_archive=None, source_sha256=None):
    plan = managed_plan(root, uv)
    root = Path(plan["persistent_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "python-executable.txt").exists():
        raise ValueError("managed Python already recorded; reuse it or choose a new root, never silently upgrade")
    env = {**os.environ, **plan["environment"]}
    source = None
    if python is None:
        if provider != "uv" or source_archive or source_sha256:
            raise ValueError("uv mode uses uv-managed installation; external provider requires --python")
        subprocess.run(plan["commands"][0], env=env, check=True, timeout=1800)
        located = subprocess.run(plan["commands"][1], env=env, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        prefix = root / "python"
    else:
        if provider not in ("conda", "uploaded"):
            raise ValueError("external Python requires provider conda or uploaded")
        located = str(python)
        prefix = root
        if provider == "uploaded":
            if not source_archive or not source_sha256 or sha256(source_archive) != source_sha256:
                raise ValueError("uploaded Python requires its official archive and matching SHA256")
            source = {"archive": str(Path(source_archive).resolve()), "sha256": source_sha256}
    executable = Path(located).resolve()
    executable.relative_to(prefix.resolve())
    code = "import json,sys; print(json.dumps({'version':list(sys.version_info[:2]),'base_prefix':sys.base_prefix,'executable':sys.executable}))"
    probe = json.loads(subprocess.run([str(executable), "-c", code], env=env, capture_output=True, text=True, check=True, timeout=30).stdout)
    if probe["version"] != [3, 11]:
        raise ValueError("managed interpreter must be Python 3.11")
    Path(probe["base_prefix"]).resolve().relative_to(prefix.resolve())
    Path(probe["executable"]).resolve().relative_to(prefix.resolve())
    (root / "python-executable.txt").write_text(str(executable) + "\n", encoding="utf-8")
    report = {**plan, "status": "managed_python_ready", "provider": provider, "source_archive": source,
              "python": probe, "resolved_executable": str(executable)}
    if python is not None:
        report["commands"] = []  # Registration never installs/creates a conda environment.
    (root / "python-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--allow-install", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=Path, help="Existing Python3.11 and base_prefix must both be inside PERSIST")
    parser.add_argument("--provider", choices=("uv", "conda", "uploaded"), default="uv")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if args.dry_run:
        plan = managed_plan(args.persistent_root, args.uv)
        if args.python:
            plan.update(commands=[], provider=args.provider, executable=str(args.python), source_archive=str(args.source_archive) if args.source_archive else None)
        print(json.dumps(plan, indent=2)); return
    if not args.allow_install:
        parser.error("managed Python download/install requires manual --allow-install")
    print(json.dumps(prepare_managed_python(args.persistent_root, args.uv, python=args.python, provider=args.provider,
                                          source_archive=args.source_archive, source_sha256=args.source_sha256), indent=2))


if __name__ == "__main__":
    main()
