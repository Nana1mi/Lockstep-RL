"""Manual uv-managed Python on persistent storage; no GPU or environment reuse."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def managed_plan(root, uv="uv"):
    root = Path(root).resolve()
    return {"status": "INSTALL_PENDING", "persistent_root": str(root),
            "environment": {"UV_PYTHON_INSTALL_DIR": str(root / "python"), "UV_CACHE_DIR": str(root / "uv-cache")},
            "commands": [[uv, "python", "install", "--no-bin", "3.11"],
                         [uv, "python", "find", "--managed-python", "3.11"]],
            "note": "Preserve the same absolute mount path/OS/architecture. Recreate venv if those change."}


def prepare_managed_python(root, uv="uv"):
    plan = managed_plan(root, uv)
    root = Path(plan["persistent_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "python-executable.txt").exists():
        raise ValueError("managed Python already recorded; reuse it or choose a new root, never silently upgrade")
    env = {**os.environ, **plan["environment"]}
    subprocess.run(plan["commands"][0], env=env, check=True)
    located = subprocess.run(plan["commands"][1], env=env, capture_output=True, text=True, check=True).stdout.strip()
    executable = Path(located).resolve()
    executable.relative_to((root / "python").resolve())
    code = "import json,sys; print(json.dumps({'version':list(sys.version_info[:2]),'base_prefix':sys.base_prefix,'executable':sys.executable}))"
    probe = json.loads(subprocess.run([str(executable), "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    if probe["version"] != [3, 11]:
        raise ValueError("managed interpreter must be Python 3.11")
    Path(probe["base_prefix"]).resolve().relative_to((root / "python").resolve())
    (root / "python-executable.txt").write_text(str(executable) + "\n", encoding="utf-8")
    report = {**plan, "status": "managed_python_ready", "python": probe, "resolved_executable": str(executable)}
    (root / "python-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--allow-install", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(managed_plan(args.persistent_root, args.uv), indent=2)); return
    if not args.allow_install:
        parser.error("managed Python download/install requires manual --allow-install")
    print(json.dumps(prepare_managed_python(args.persistent_root, args.uv), indent=2))


if __name__ == "__main__":
    main()
