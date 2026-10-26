"""Bind CPU evidence to normalized source text and verify the delivered Git tree."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "results/validation-manifest.json"
    if args.verify_git:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        for name, expected in manifest["source_files"].items():
            result = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, check=True)
            if digest(result.stdout) != expected:
                raise ValueError(f"committed source differs from validated text: {name}")
        print(f"Verified {len(manifest['source_files'])} source files against delivered HEAD")
        return
    log = (ROOT / "results/cpu-tests.txt").read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests in ([0-9.]+)s", log)
    if not count or not log.rstrip().endswith("OK"):
        raise ValueError("CPU test log does not record a complete passing run")
    for filename in ("cpu-artifact-checks.json", "package-check.json"):
        if json.loads((ROOT / "results" / filename).read_text())["status"] != "passed":
            raise ValueError(f"failed validation artifact: {filename}")
    if "All checks passed!" not in (ROOT / "results/static-checks.txt").read_text():
        raise ValueError("static check result missing")
    files = [ROOT / "pyproject.toml"]
    for directory, pattern in (("nanovllm", "*.py"), ("lockstep_rl", "*.py"), ("scripts", "*.py"),
                               ("scripts", "*.sh"), ("tests", "*.py"), ("configs", "*.json")):
        files.extend(sorted((ROOT / directory).rglob(pattern)))
    files.extend([ROOT / "example.py", ROOT / "bench.py"])
    source = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path.read_bytes()) for path in files}
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "cpu_tests": int(count.group(1)), "cpu_test_seconds": float(count.group(2)),
              "source_normalization": "UTF-8 bytes with CRLF normalized to LF, matching Git text normalization",
              "source_files": source, "gpu_validation": "GPU_PENDING",
              "scope": "CPU evidence and committed source identity; no real GPU result"}
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Bound {count.group(1)} passing CPU tests to {len(source)} source/config files")


if __name__ == "__main__":
    main()
