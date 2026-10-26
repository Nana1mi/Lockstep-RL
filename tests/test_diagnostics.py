import copy
import unittest
from types import SimpleNamespace
from nanovllm.engine.scheduler import Scheduler

from lockstep_rl.diagnostics import compare_trajectories, distribution, path_evidence, require_path
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tests.test_rollout import cpu_engine


class DiagnosticsTest(unittest.TestCase):
    def test_six_block_preemption_cache_switch_and_score_alignment(self):
        original = Sequence.block_size
        Sequence.block_size = 256
        try:
            evidence = {}
            for enabled in (True, False):
                scheduler = Scheduler(SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=2048,
                    eos=-1, kvcache_block_size=256, num_kvcache_blocks=6,
                    record_trace=True, enable_prefix_caching=enabled))
                sequences = [Sequence([100 + i] * 512, SamplingParams(max_tokens=300, ignore_eos=True),
                                      request_id=f"traj-{i}") for i in range(2)]
                for seq in sequences:
                    scheduler.add(seq)
                for _ in range(2000):
                    if scheduler.is_finished():
                        break
                    seqs, prefill = scheduler.schedule()
                    tokens = [1000 + seq.num_completion_tokens for seq in seqs]
                    scores = [-float(token) / 1000 for token in tokens]
                    scheduler.postprocess(seqs, tokens, prefill, scores)
                self.assertTrue(scheduler.is_finished())
                case = "preemption_cached" if enabled else "preemption_recompute"
                evidence[enabled] = require_path(scheduler.trace, case)
                for seq in sequences:
                    self.assertEqual(seq.completion_token_ids, list(range(1000, 1300)))
                    self.assertEqual(seq.response_logprobs, [-t / 1000 for t in range(1000, 1300)])
                    self.assertEqual(seq.loss_mask, [1] * 300)
                if not enabled:
                    self.assertFalse(scheduler.block_manager.hash_to_block_id)
            self.assertGreater(evidence[True]["prefix_reused_tokens"], 0)
            self.assertEqual(evidence[False]["prefix_reused_tokens"], 0)
        finally:
            Sequence.block_size = original

    def test_multi_trajectory_metrics_are_token_weighted_but_threshold_counts_trajectories(self):
        rows = [{"request_id": f"traj-{i}", "prompt_ids": [1], "response_ids": [2] * n,
                 "response_logprobs": [-2.] * n} for i, n in enumerate((1, 3, 2))]
        other = copy.deepcopy(rows)
        other[0]["response_logprobs"] = [-1.6]
        other[1]["response_logprobs"] = [-1.9] * 3
        other[2]["response_logprobs"] = [-1.8, -1.4]
        stats = compare_trajectories(rows, other, threshold=.3)
        self.assertEqual(stats["trajectory_count"], 3)
        self.assertEqual(stats["trajectories_above_threshold"], 2)
        self.assertAlmostEqual(stats["absolute_logprob_error"]["mean"], 1.5 / 6)
        self.assertEqual([round(r["max_absolute_logprob_error"], 3) for r in stats["per_trajectory"]], [.4, .1, .6])
        self.assertEqual(stats["rows"][0]["rollout_logprob"], -2.)
        self.assertEqual(stats["rows"][0]["reference_logprob"], -1.6)

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
        trace = [{"event": "schedule", "is_prefill": True, "requests": [
                     {"request_id": "r", "start": 0, "end": 7, "context_tokens": 7}]},
                 {"event": "preempt", "request_id": "r"},
                 {"event": "schedule", "is_prefill": True, "requests": [
                     {"request_id": "r", "start": 4, "end": 8, "context_tokens": 8}]},
                 {"event": "execution", "is_prefill": False, "actual_batch": 3,
                  "graph_batch": 4, "graph_replay": True}]
        self.assertEqual(require_path(trace, "preemption")["re_prefill_requests"], ["r"])
        self.assertEqual(require_path(trace, "graph_padding")["padded_graph_replays"], 1)
        self.assertEqual(path_evidence(trace)["chunked_request_count"], 0)
        self.assertEqual(path_evidence([])["preemptions"], 0)


if __name__ == "__main__":
    unittest.main()
