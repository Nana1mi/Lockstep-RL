"""Manual read-only L40S preflight; no CUDA imports, installations or downloads."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lockstep_rl.experiment_io import save_json
from lockstep_rl.rental import data_mount, validate_gpu
from scripts.run_rental import gpu_inventory

DOMAINS = ("pypi.org", "download.pytorch.org", "github.com", "huggingface.co")


def network_probe():
    results = {}
    for domain in DOMAINS:
        command = ["curl", "-I", "--connect-timeout", "5", "--max-time", "20", f"https://{domain}"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=25)
            results[domain] = {"command": command, "returncode": result.returncode,
                               "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            results[domain] = {"command": command, "error": f"{type(exc).__name__}: {exc}"}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", help="Exact name manually confirmed from nvidia-smi")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--region", default="安徽区", help="Operator-confirmed platform region")
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/rental.json").read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps({"status": "PREFLIGHT_PENDING", "gpu_name": None, "driver": None, "mount": None,
                          "network": dict.fromkeys(DOMAINS), "minimum_driver_major": settings["minimum_driver_major"]}, indent=2))
        return
    expected = args.expected_gpu_name or settings["expected_gpu_name"]
    inventory = gpu_inventory(args.gpu_id)
    validate_gpu(inventory, expected, settings)
    if args.region != settings["platform"]["region"]:
        parser.error("confirm the selected Anhui region; do not mix regions")
    mount = data_mount(args.persistent_root)
    output = args.output or args.persistent_root / "rental-preflight.json"
    output.resolve().relative_to(args.persistent_root.resolve())
    if output.exists():
        parser.error("preflight record exists; preserve it and choose a new output path")
    record = {"status": "confirmed", "confirmed_at": datetime.now(timezone.utc).isoformat(),
              "expected_gpu_name": expected, "gpu": inventory, "mount": mount, "region": args.region,
              "platform": settings["platform"], "network": network_probe() if args.check_network else dict.fromkeys(DOMAINS),
              "gpu_validation": "GPU_PENDING", "scope": "read-only metadata; network failures require explicit fallback selection"}
    save_json(output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
