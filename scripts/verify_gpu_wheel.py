"""Check the pinned official FA2 wheel and local torch ABI without GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_wheel(path, settings, *, torch_version, torch_cuda, cxx11_abi, python_version, machine):
    path = Path(path)
    if path.name != settings["wheel_filename"]:
        raise ValueError("wheel filename does not match pinned torch/CUDA/Python/ABI candidate")
    if python_version != (3, 11) or machine != "x86_64":
        raise ValueError("rental wheel requires Linux x86_64 / Python 3.11")
    if torch_version.split("+")[0] != "2.8.0" or not str(torch_cuda).startswith("12.") or not cxx11_abi:
        raise ValueError("wheel requires torch 2.8.0 CUDA 12 / CXX11 ABI TRUE")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != settings["wheel_sha256"]:
        raise ValueError("wheel SHA256 differs from pinned official release asset")
    return {"status": "passed", "wheel": str(path), "sha256": digest.hexdigest(),
            "scope": "binary metadata/hash only; CUDA execution remains GPU_PENDING"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("rental candidate is Linux only")
    import torch
    settings = json.loads((ROOT / "configs/rental.json").read_text())
    report = verify_wheel(args.wheel, settings, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                          cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                          python_version=sys.version_info[:2], machine=platform.machine())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
