"""Linux process/port regressions, runnable with python -S and no PyTorch."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lockstep_rl.process_runner import (
    TaskResult, execute_worker_jobs, run_task, task_budget,
)
from lockstep_rl.rendezvous import check_port, configured_port, rendezvous_settings


@unittest.skipUnless(sys.platform.startswith("linux"), "process-tree cleanup targets Linux/WSL2")
class ProcessRunnerTests(unittest.TestCase):
    def task(self, code, *, timeout=3, **kwargs):
        return run_task([sys.executable, "-S", "-c", code], timeout_seconds=timeout,
                        grace_seconds=0.1, **kwargs)

    def assert_gone(self, pid):
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_normal_exit_preserves_code_and_restores_signal_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        result = self.task("raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.cleanup["surviving_pids"], [])
        self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})

    def test_timeout_kills_child_and_records_log(self):
        # Popen requires a real descriptor for stdout.
        with tempfile.TemporaryFile(mode="w+") as log:
            result = self.task("import time; time.sleep(60)", timeout=0.25, stdout=log)
            log.seek(0)
            self.assertIn('"timed_out": true', log.read())
        self.assertEqual(result.returncode, 124)
        self.assertTrue(result.timed_out)
        self.assertLess(result.wall_seconds, 3)
        self.assert_gone(result.pid)

    def test_term_ignoring_descendant_in_separate_session_is_killed_and_port_released(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.json"
            grandchild = (
                "import json,os,signal,socket,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
                f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),s.getsockname()[1]])); "
                "time.sleep(60)"
            )
            code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-S','-c',{grandchild!r}],start_new_session=True); time.sleep(60)"
            result = self.task(code, timeout=1)
            pid, port = json.loads(marker.read_text())
            self.assertTrue(result.timed_out)
            self.assertTrue(result.cleanup["sigkill_required"])
            self.assertGreaterEqual(len(result.cleanup["process_groups"]), 2)
            self.assert_gone(pid)
            self.assertEqual(check_port(port), port)

    def test_fast_parent_exit_does_not_leave_detached_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pid"
            child = "import time; time.sleep(60)"
            code = ("import subprocess,sys; from pathlib import Path; "
                    f"p=subprocess.Popen([sys.executable,'-S','-c',{child!r}],start_new_session=True); "
                    f"Path({str(marker)!r}).write_text(str(p.pid))")
            result = self.task(code)
            self.assertEqual(result.returncode, 0)
            self.assert_gone(int(marker.read_text()))

    def test_nested_runner_interruption_cleans_its_own_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "leaf"
            leaf = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
            nested = ("import sys; from lockstep_rl.process_runner import run_task; "
                      f"run_task([sys.executable,'-S','-c',{leaf!r}],timeout_seconds=20,grace_seconds=.1)")
            result = self.task(nested, timeout=1)
            self.assertTrue(result.timed_out)
            self.assert_gone(int(marker.read_text()))

    def test_sigterm_of_supervisor_cleans_workers_before_interrupt_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "leaf"
            leaf = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
            code = ("import sys; from lockstep_rl.process_runner import run_task; "
                    f"run_task([sys.executable,'-S','-c',{leaf!r}],timeout_seconds=20,grace_seconds=.1)")
            supervisor = subprocess.Popen([sys.executable, "-S", "-c", code], stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(marker.exists())
                supervisor.send_signal(signal.SIGTERM)
                self.assertNotEqual(supervisor.wait(timeout=5), 0)
                self.assert_gone(int(marker.read_text()))
            finally:
                if supervisor.poll() is None:
                    os.killpg(supervisor.pid, signal.SIGKILL)
                    supervisor.wait(timeout=5)

    def test_parent_deadline_limits_child_budget_without_starting_expired_work(self):
        env = {**os.environ, "LOCKSTEP_TASK_DEADLINE": str(time.time() - 1)}
        result = self.task("raise RuntimeError('must not execute')", env=env)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.pid)
        env["LOCKSTEP_TASK_DEADLINE"] = str(time.time() + .3)
        result = self.task("import time; time.sleep(60)", env=env)
        self.assertTrue(result.timed_out)
        self.assertLess(result.timeout_seconds, .31)

    def test_port_conflict_never_starts_child_or_kills_owner(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            port = listener.getsockname()[1]
            env = {**os.environ, "LOCKSTEP_NCCL_PORT": str(port)}
            result = self.task("raise RuntimeError('must not execute')", nccl=True, env=env)
            self.assertEqual(result.returncode, 1)
            self.assertIsNone(result.pid)
            self.assertIn("OSError", result.error)
            self.assertEqual(listener.getsockname()[1], port)
        self.assertEqual(check_port(port), port)

    def test_time_wait_is_not_mistaken_for_a_live_listener(self):
        # Match a reusable TCP rendezvous listener, then make its accepted
        # connection close first so the server port enters TIME_WAIT.
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0)); server.listen()
            port = server.getsockname()[1]
            client.connect(("127.0.0.1", port))
            accepted, _ = server.accept()
            accepted.close()
            self.assertEqual(client.recv(1), b"")
        self.assertEqual(check_port(port), port)

    def test_auto_port_and_explicit_override_reach_child(self):
        for port in (None, check_port(0)):
            env = {key: value for key, value in os.environ.items() if key != "LOCKSTEP_NCCL_PORT"}
            if port:
                env["LOCKSTEP_NCCL_PORT"] = str(port)
            with tempfile.TemporaryFile(mode="w+") as log:
                result = self.task("import os; print(os.environ['LOCKSTEP_NCCL_PORT'])", nccl=True, env=env, stdout=log)
                log.seek(0)
                observed = int(log.readline())
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.nccl_port, observed)
            if port:
                self.assertEqual(port, observed)

    def test_original_baseline_checks_fixed_port_even_with_override(self):
        with patch("lockstep_rl.process_runner.check_port", side_effect=OSError("occupied")) as check:
            result = self.task("pass", nccl="baseline", env={"LOCKSTEP_NCCL_PORT": "30000"})
        check.assert_called_once_with(2333, dual_stack=True)
        self.assertEqual(result.returncode, 1)

    def test_baseline_localhost_preflight_detects_ipv6_listener(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 unavailable")
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                listener.bind(("::1", 0))
            except OSError:
                self.skipTest("IPv6 loopback unavailable")
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                check_port(port, dual_stack=True)
            self.assertEqual(listener.getsockname()[1], port)

    def test_inherited_automatic_port_selects_fresh_free_port(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            occupied = listener.getsockname()[1]
            env = {**os.environ, "LOCKSTEP_NCCL_PORT": str(occupied), "LOCKSTEP_NCCL_AUTO_PORT": "1"}
            result = self.task("pass", nccl=True, env=env)
            self.assertEqual(result.returncode, 0)
            self.assertNotEqual(result.nccl_port, occupied)

    def test_worker_timeout_continues_and_sample_failure_skips_dependents(self):
        jobs = [{"name": "sample", "command": ["fake"], "timeout_seconds": 3},
                {"name": "bad", "command": ["fake"], "timeout_seconds": 3, "depends_on": ["sample"]},
                {"name": "good", "command": ["fake"], "timeout_seconds": 3, "depends_on": ["sample"]}]
        for results, expected_calls in (([TaskResult(returncode=0), TaskResult(returncode=124, timed_out=True), TaskResult(returncode=0)], 3),
                                        ([TaskResult(returncode=124, timed_out=True)], 1)):
            with tempfile.TemporaryDirectory() as directory:
                manifest = {"jobs": []}
                with patch("lockstep_rl.process_runner.run_task", side_effect=results) as process:
                    with self.assertRaises(RuntimeError):
                        execute_worker_jobs(jobs, directory, manifest)
                self.assertEqual(process.call_count, expected_calls)
                self.assertEqual(len(manifest["jobs"]), 3)
                self.assertEqual(manifest["jobs"][-1]["status"], "completed" if expected_calls == 3 else "skipped_dependency")
                self.assertEqual(json.loads((Path(directory) / "manifest.json").read_text()), manifest)

    def test_invalid_limits_and_launch_failure(self):
        for limit in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.task("pass", timeout=limit)
        result = run_task(["/nonexistent/lockstep-test-command"], timeout_seconds=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FileNotFoundError", result.error)

    def test_rental_real_timeout_continues_but_environment_timeout_stops(self):
        from scripts.run_rental import execute_jobs

        inventory = {"name": "cpu-fixture"}
        for hard_gate in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                jobs = [{"name": "hang", "hard_gate": hard_gate, "asset_splits": [],
                         "timeout_seconds": .2, "command": [sys.executable, "-S", "-c", "import time; time.sleep(60)"]},
                        {"name": "next", "hard_gate": False, "asset_splits": [],
                         "timeout_seconds": 3, "command": [sys.executable, "-S", "-c", "pass"]}]
                plan = {"output": directory, "jobs": jobs}
                report = execute_jobs(plan, inventory, dict(os.environ), "0", validate_inputs=False,
                                      inventory_reader=lambda _: inventory)
                self.assertEqual(report["failed_jobs"], ["hang"])
                self.assertTrue(report["completed_jobs"][0]["timed_out"])
                self.assertEqual(len(report["completed_jobs"]), 1 if hard_gate else 2)
                self.assertEqual(report["status"], "blocked_by_environment" if hard_gate else "completed_with_failures")


class RendezvousTests(unittest.TestCase):
    def test_configuration_is_validated_without_torch(self):
        self.assertEqual(configured_port({}), 2333)
        url, timeout = rendezvous_settings({"LOCKSTEP_NCCL_PORT": "23456", "LOCKSTEP_NCCL_INIT_TIMEOUT_SECONDS": "9"})
        self.assertEqual(url, "tcp://127.0.0.1:23456")
        self.assertEqual(timeout.total_seconds(), 9)
        for value in ("0", "65536", "nonsense"):
            with self.assertRaises(ValueError):
                configured_port({"LOCKSTEP_NCCL_PORT": value})
        for value in ("0", "nan", "inf"):
            with self.assertRaises(ValueError):
                rendezvous_settings({"LOCKSTEP_NCCL_INIT_TIMEOUT_SECONDS": value})

    def test_timeout_budgets_cover_tasks_and_are_three_times_estimates(self):
        for name in ("environment", "attention", "baseline", "diagnostics", "weight-sync", "two-updates",
                     "smoke-rollout", "smoke-recomputed", "train-demo-simple", "train-demo-simple-lr5e-6",
                     "gsm8k-smoke", "performance", "sampler", "diagnostic-worker", "sync-worker", "benchmark-worker", "plot", "archive"):
            budget = task_budget(name)
            self.assertGreater(budget["expected_seconds"], 0)
            self.assertEqual(budget["timeout_seconds"], 3 * budget["expected_seconds"])


if __name__ == "__main__":
    unittest.main()
