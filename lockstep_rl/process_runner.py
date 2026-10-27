"""Bounded Linux task execution, including nested sessions and orphan cleanup.

Only used by serial, dedicated orchestration processes on their main thread.
No torch import, CUDA initialization, model construction or download occurs here.
"""

import ctypes
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from lockstep_rl.rendezvous import check_port, configured_port


class ProcessCleanupError(BaseException):
    """Infrastructure failure: do not start another GPU job beside survivors."""


class TaskInterrupted(KeyboardInterrupt):
    pass


@dataclass
class TaskResult:
    returncode: int = 1
    timed_out: bool = False
    error: str | None = None
    timeout_seconds: float = 0
    wall_seconds: float = 0
    pid: int | None = None
    process_group: int | None = None
    nccl_port: int | None = None
    cleanup: dict = field(default_factory=dict)

    def record(self):
        return asdict(self)


def task_budget(kind):
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/task-timeouts.json").read_text())
    expected = settings["expected_seconds"][kind]
    return {"expected_seconds": expected, "timeout_seconds": expected * settings["multiplier"],
            "timeout_basis": "unmeasured expected wall seconds times multiplier"}


def _table():
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = (int(fields[1]), int(fields[2]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            continue  # Process exited during enumeration.
    return rows


def _subreaper():
    # Linux man-pages PR_SET_CHILD_SUBREAPER: adopt even double-forked/setsid
    # descendants, so a fast-exiting worker cannot hide an orphan from cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
    return libc, previous.value


_active = threading.Lock()


def run_task(command, *, timeout_seconds, stdout=None, stderr=subprocess.STDOUT,
             cwd=None, env=None, nccl=False, grace_seconds=5):
    """Return failed/timeout records; interrupts propagate after tree cleanup.

    nccl=True checks an explicit override, or selects a free loopback port.
    nccl='baseline' always checks immutable upstream's port 2333. The bind is
    a preflight, not a reservation; the finite rendezvous/task deadlines also
    cover the race between preflight and the real listener starting.
    """
    if not sys.platform.startswith("linux") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("managed tasks require Linux/WSL2 and the main thread")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("task timeout must be positive; cleanup grace must be nonnegative")
    if not _active.acquire(blocking=False):
        raise RuntimeError("run_task is serial; concurrent child ownership is unsupported")
    start = time.monotonic()
    result = TaskResult(timeout_seconds=timeout_seconds)
    process = None
    libc = None
    handlers = {}
    tracked = {}  # pid -> /proc start time: never signal a recycled PID.
    before_children = set()
    cleanup_started = False

    def discover():
        rows = _table()
        owned = {pid for pid, row in rows.items() if tracked.get(pid) == row[3]}
        if process and process.pid in rows:
            owned.add(process.pid)
        owned.update(pid for pid, row in rows.items() if row[0] == os.getpid() and pid not in before_children)
        while True:
            more = {pid for pid, row in rows.items() if row[0] in owned} - owned
            if not more:
                break
            owned.update(more)
        for pid in owned:
            tracked[pid] = rows[pid][3]
        return {pid: rows[pid] for pid in owned if pid in rows}

    def cleanup():
        nonlocal cleanup_started
        cleanup_started = True
        # Ignore repeated termination while carrying out bounded cleanup.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        signaled_groups = set()
        killed = False

        def send(rows, sig):
            groups = {row[1] for row in rows.values()}
            if process:
                groups.add(process.pid)  # includes descendants after leader exit
            groups.discard(os.getpgrp())
            for pgid in groups:
                try:
                    os.killpg(pgid, sig)
                    signaled_groups.add(pgid)
                except ProcessLookupError:
                    pass
            # Also cover a descendant which changed its group after the scan.
            for pid, row in rows.items():
                try:
                    if _table().get(pid, (None,) * 4)[3] == row[3]:
                        os.kill(pid, sig)
                except ProcessLookupError:
                    pass

        def reap():
            process.poll()  # Popen owns/reaps its direct child.
            rows = discover()
            for pid, row in rows.items():
                if pid != process.pid and row[0] == os.getpid():
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            return discover()

        rows = discover()
        send(rows, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while rows and time.monotonic() < deadline:
            time.sleep(0.05)
            rows = reap()
            send(rows, signal.SIGTERM)
        if rows:
            killed = True
            deadline = time.monotonic() + 5
            while rows and time.monotonic() < deadline:
                send(rows, signal.SIGKILL)
                time.sleep(0.05)
                rows = reap()
        result.cleanup = {"process_groups": sorted(signaled_groups), "sigkill_required": killed,
                          "surviving_pids": sorted(rows), "subreaper": True}
        if rows:
            raise ProcessCleanupError(f"task descendants survived SIGKILL; stop rental execution: {sorted(rows)}")
        process.wait(timeout=1)

    try:
        child_env = dict(os.environ if env is None else env)
        # Inherited absolute deadline bounds nested work and in-process setup.
        deadline = min(time.time() + timeout_seconds, float(child_env.get("LOCKSTEP_TASK_DEADLINE", "inf")))
        result.timeout_seconds = max(0, deadline - time.time())
        local_deadline = time.monotonic() + result.timeout_seconds
        if not math.isfinite(deadline):
            raise ValueError("task deadline must be finite")
        child_env["LOCKSTEP_TASK_DEADLINE"] = str(deadline)
        if nccl:
            automatic = "LOCKSTEP_NCCL_PORT" not in child_env or child_env.get("LOCKSTEP_NCCL_AUTO_PORT") == "1"
            port = 2333 if nccl == "baseline" else 0 if automatic else configured_port(child_env)
            result.nccl_port = check_port(port, dual_stack=True) if nccl == "baseline" else check_port(port)
            child_env["LOCKSTEP_NCCL_PORT"] = str(result.nccl_port)
            if automatic:
                # Sequential workers should not reuse the previous listener's
                # TIME_WAIT port. All ranks inside this worker still share it.
                child_env["LOCKSTEP_NCCL_AUTO_PORT"] = "1"
        if result.timeout_seconds <= 0:
            result.timed_out = True
            result.returncode = 124
            result.error = "inherited task deadline already expired"
            return result
        libc, previous_subreaper = _subreaper()
        before_children = {pid for pid, row in _table().items() if row[0] == os.getpid()}

        def interrupted(sig, frame):
            raise TaskInterrupted(f"received signal {sig}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        process = subprocess.Popen(command, cwd=cwd, env=child_env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        result.pid = result.process_group = process.pid
        while True:
            discover()
            code = process.poll()
            if code is not None:
                result.returncode = code
                break
            if time.monotonic() >= local_deadline:
                result.returncode = 124
                result.timed_out = True
                result.error = "task wall-clock timeout"
                break
            time.sleep(min(0.05, max(0, local_deadline - time.monotonic())))
    except Exception as exc:
        result.returncode = 1
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if process and not cleanup_started:
                cleanup()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if libc:
                libc.prctl(36, previous_subreaper, 0, 0, 0)
            result.wall_seconds = time.monotonic() - start
            _active.release()
            if stdout is not None and hasattr(stdout, "write"):
                try:
                    stdout.write("\n[lockstep-task] " + json.dumps(result.record()) + "\n")
                    stdout.flush()
                except (TypeError, OSError):
                    pass  # Binary git archive output must remain byte-for-byte intact.
    return result


def run_checked(command, *, kind="plot", **kwargs):
    result = run_task(command, timeout_seconds=task_budget(kind)["timeout_seconds"], **kwargs)
    if result.returncode:
        raise RuntimeError(f"subprocess failed: {result.record()}")
    return result


def execute_worker_jobs(jobs, output, manifest):
    """Run independent workers despite failures; skip missing prerequisites.

    The caller only computes a complete suite summary if every worker succeeds.
    Each worker's raw output and failure/timeout record survives partial suites.
    """
    from lockstep_rl.experiment_io import save_json

    output = Path(output)
    successful = set()
    for job in jobs:
        missing = set(job.get("depends_on", [])) - successful
        log_path = output / f"{job['name']}.log"
        if missing:
            record = {**job, "status": "skipped_dependency", "returncode": None,
                      "reason": f"prerequisite failed: {sorted(missing)}"}
        else:
            try:
                with log_path.open("w", encoding="utf-8") as stream:
                    result = run_task(job["command"], timeout_seconds=job["timeout_seconds"], stdout=stream,
                                      nccl="baseline" if job.get("source") == "baseline" else True)
                record = {**job, **result.record(), "log": str(log_path),
                          "status": "timed_out" if result.timed_out else "failed" if result.returncode else "completed"}
                if not result.returncode:
                    successful.add(job["name"])
            except BaseException as exc:
                manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_failure",
                                error=f"{type(exc).__name__}: {exc}")
                save_json(output / "manifest.json", manifest)
                raise
        manifest["jobs"].append(record)
        save_json(output / "manifest.json", manifest)
    failures = [job["name"] for job in jobs if job["name"] not in successful]
    if failures:
        raise RuntimeError(f"incomplete suite; failed/timed-out/skipped workers: {failures}; inspect saved logs/manifest")
