"""External orchestration guard; experimental source stays at ce9e354."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

EXPECTED = "ce9e3542b3f31475ffa595e3f09b80e3f9190ed0"


def identity(repo):
    def git(*args):
        return subprocess.run(["git", "-c", f"safe.directory={repo}", *args],
                              cwd=repo, capture_output=True, text=True,
                              check=True, timeout=30).stdout
    actual = git("rev-parse", "HEAD").strip()
    if actual != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED}; found {actual}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty.strip():
        raise RuntimeError("Tracked worktree/index must be clean: " + dirty)
    return {"commit": actual, "tracked_worktree_clean": True,
            "untracked_files": git("ls-files", "--others", "--exclude-standard").splitlines(),
            "timeout_config_sha256": hashlib.sha256(
                (repo / "configs/task-timeouts.json").read_bytes()).hexdigest(),
            "external_launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Pinned experimental source; explicit orchestration override below"}


def annotate(plan, source, performance_cap):
    plan["source_commit"] = source["commit"]
    plan["source"] = source
    plan["timeout_overrides"] = []
    for job in plan["jobs"]:
        if job["name"] == "performance":
            previous = job["timeout_seconds"]
            if performance_cap < previous:
                raise ValueError("Performance cap may only increase")
            job["timeout_seconds"] = performance_cap
            job["timeout_basis"] = "Explicit external cap; 96-worker matrix unmeasured; short preflight extrapolation exceeds half the original parent cap"
            plan["timeout_overrides"].append({"job": "performance",
                "original_cap_seconds": previous, "runtime_cap_seconds": performance_cap,
                "source_config_unchanged": True})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--performance-cap-seconds", type=float, default=12000)
    parser.add_argument("rental_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = identity(repo)  # Happens before importing any project code.
    if not 12000 <= args.performance_cap_seconds <= 86400:
        parser.error("performance cap must be between12000 and86400 seconds")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("pinned_run_rental", repo / "scripts/run_rental.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_plan, original_execute = module.build_plan, module.execute_jobs
    def build(*a, **kw):
        return annotate(original_plan(*a, **kw), source, args.performance_cap_seconds)
    def execute(plan, inventory, env, gpu_id, **kwargs):
        supplied_reader = kwargs.pop("inventory_reader", None) or module.gpu_inventory
        def guarded_inventory(selected_gpu):
            identity(repo)  # Verify pinned clean source again before each job.
            return supplied_reader(selected_gpu)
        identity(repo)
        result = original_execute(plan, inventory, env, gpu_id,
                                  inventory_reader=guarded_inventory, **kwargs)
        identity(repo)
        return result
    module.build_plan, module.execute_jobs = build, execute
    forwarded = args.rental_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = [str(repo / "scripts/run_rental.py"), *forwarded]
    module.main()


if __name__ == "__main__":
    main()
