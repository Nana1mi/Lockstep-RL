import copy
import unittest

from lockstep_rl.diagnostics import compare_trajectories, distribution, path_evidence, require_path
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tests.test_rollout import cpu_engine


class DiagnosticsTest(unittest.TestCase):
    def test_fixed_trajectory_statistics_and_known_error_label(self):
        record = {"request_id": "r", "prompt_ids": [1, 2], "response_ids": [3, 4],
                  "response_logprobs": [-1., -2.]}
        other = {**record, "response_logprobs": [-.9, -2.2]}
        result = compare_trajectories([record], [other], threshold=.15)
        self.assertAlmostEqual(result["absolute_logprob_error"]["mean"], .15)
        self.assertEqual(result["above_threshold_fraction"], .5)
        self.assertFalse(result["known_error_injection"])
        self.assertTrue(compare_trajectories([record], [other], known_error=True)["known_error_injection"])
        wrong = copy.deepcopy(other); wrong["response_ids"] = [4, 3]
        with self.assertRaises(ValueError):
            compare_trajectories([record], [wrong])
        with self.assertRaises(ValueError):
            distribution([float("nan")])

    def test_cpu_scheduler_records_actual_prefix_and_chunk_ranges(self):
        old_size = Sequence.block_size
        Sequence.block_size = 4
        try:
            engine = cpu_engine(budget=3)
            engine.scheduler.record_trace = True
            engine.generate([list(range(10))], SamplingParams(max_tokens=1), use_tqdm=False)
            evidence = require_path(engine.scheduler.trace, "chunked")
            ranges = next(iter(evidence["prefill_ranges"].values()))
            self.assertEqual(ranges, [(0, 3), (3, 6), (6, 9), (9, 10)])
            engine.scheduler.trace.clear()
            engine.generate([list(range(10))], SamplingParams(max_tokens=1), use_tqdm=False)
            self.assertEqual(require_path(engine.scheduler.trace, "prefix")["prefix_reused_tokens"], 8)
            with self.assertRaises(RuntimeError):
                require_path(engine.scheduler.trace, "graph")
        finally:
            Sequence.block_size = old_size

    def test_graph_and_preemption_evidence_requires_actual_events(self):
        trace = [{"event": "preempt", "request_id": "r"},
                 {"event": "schedule", "is_prefill": True, "requests": [
                     {"request_id": "r", "start": 0, "end": 7}]},
                 {"event": "execution", "is_prefill": False, "actual_batch": 3,
                  "graph_batch": 4, "graph_replay": True}]
        self.assertEqual(require_path(trace, "preemption")["re_prefill_requests"], ["r"])
        self.assertEqual(require_path(trace, "graph_padding")["padded_graph_replays"], 1)
        self.assertEqual(path_evidence([])["preemptions"], 0)


if __name__ == "__main__":
    unittest.main()
