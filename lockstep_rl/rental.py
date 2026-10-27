"""Standard-library-only rental identity, persistent mount and hash checks."""

import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root, paths):
    root = Path(root)
    return {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(paths)}


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    if not hashes:
        raise ValueError("asset file hashes are missing")
    for name, record in hashes.items():
        path = (root / name).resolve()
        path.relative_to(root)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"asset SHA256/size mismatch: {name}")


def validate_gpu(inventory, expected_name, settings):
    if not expected_name or expected_name.startswith("USER_MUST"):
        raise ValueError("confirm the exact GPU name from nvidia-smi; expected_gpu_name is unset")
    if inventory["name"] != expected_name or not re.search(r"\bL40S\b", inventory["name"]):
        raise ValueError("actual GPU name does not match the confirmed L40S rental")
    match = re.fullmatch(r"(\d+)(?:\.\d+)+", inventory["driver"])
    if not match or int(match.group(1)) < settings["minimum_driver_major"]:
        raise ValueError("rental requires NVIDIA driver major >=570; stop without upgrading it")


def data_mount(root):
    def mount(path):
        result = subprocess.run(["findmnt", "--json", "--target", str(path), "--output",
                                 "TARGET,SOURCE,FSTYPE,MAJ:MIN"], capture_output=True, text=True,
                                check=True, timeout=15)
        return json.loads(result.stdout)["filesystems"][0]
    root = Path(root).resolve()
    found, system = mount(root), mount("/")
    if found["target"] == "/" or found["maj:min"] == system["maj:min"]:
        raise ValueError("PERSIST must be on a separate data disk, not the 30GB system filesystem")
    return {"persistent_root": str(root), "data": found, "system": system}


def validate_preflight(record, root, inventory, settings, *, mount_reader=data_mount):
    if record.get("status") != "confirmed" or record.get("region") != settings["platform"]["region"]:
        raise ValueError("confirmed preflight from the selected Anhui rental region is required")
    expected = settings.get("expected_gpu_name") or record.get("expected_gpu_name")
    validate_gpu(inventory, expected, settings)
    if record["gpu"] != inventory:
        raise ValueError("GPU/driver differs from confirmed preflight; do not mix contrasts")
    actual_mount = mount_reader(root)
    if record["mount"] != actual_mount:
        raise ValueError("data-disk mount changed after preflight")
    return record
