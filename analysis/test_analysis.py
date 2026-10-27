import math
import unittest

from analyze_l40s import paired_binary, process_statistics, quantile, sync_statistics, token_statistics
from prepare_followups import configurations


class AnalysisTest(unittest.TestCase):
    def test_linear_quantile_and_primary_token_statistics(self):
        self.assertAlmostEqual(quantile([0, 1, 2, 3], .99), 2.97)
        rows = [{'request_id': name, 'response_index': i, 'token_id': i + 3,
                 'segment': 'primary', 'delta_logp': delta}
                for name, i, delta in [('short', 0, 0), ('long', 0, .19), ('long', 1, -.23)]]
        stats = token_statistics(rows)
        self.assertEqual(stats['tokens'], 3)
        self.assertEqual(stats['trajectories_above_0_1'], 1)
        self.assertAlmostEqual(stats['mean_absolute_error'], .14)
        self.assertEqual(stats['ratio_outside_0_8_1_2'], 2 / 3)
        rows[0]['segment'] = 'post_eos'
        with self.assertRaises(ValueError):
            token_statistics(rows)

    def test_duplicate_positions_rejected_even_if_token_id_differs(self):
        rows = [{'request_id': 'x', 'response_index': 0, 'token_id': t,
                 'segment': 'primary', 'delta_logp': 0} for t in (1, 2)]
        with self.assertRaises(ValueError):
            token_statistics(rows)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            token_statistics([{'request_id': 'x', 'response_index': 0, 'token_id': 1,
                               'segment': 'primary', 'delta_logp': math.nan}])

    def test_paired_outcomes_match_by_id_and_not_row_order(self):
        left = [{'id': str(i), 'reward': v} for i, v in enumerate([1, 0, 0, 1])]
        right = [{'id': str(i), 'reward': v} for i, v in enumerate([1, 1, 0, 0])][::-1]
        result = paired_binary(left, right)
        self.assertEqual((result['gained'], result['lost']), (1, 1))
        self.assertEqual(result['exact_mcnemar_two_sided_p'], 1)
        self.assertEqual(result['difference'], 0)
        with self.assertRaises(ValueError):
            paired_binary(left, right[:-1])

    def test_exact_discordant_pair_probability(self):
        left = [{'id': str(i), 'reward': 0} for i in range(5)]
        right = [{'id': str(i), 'reward': 1} for i in range(5)]
        self.assertEqual(paired_binary(left, right)['exact_mcnemar_two_sided_p'], 2 / 32)

    def test_steady_repeats_are_nested_not_independent_processes(self):
        rows = [{'process_repeat': str(p), 'output_tokens_per_second': str(v)}
                for p, vals in [(0, [1, 3]), (1, [3, 5]), (2, [5, 7])] for v in vals]
        stats = process_statistics(rows)
        self.assertEqual(stats['independent_processes'], 3)
        self.assertEqual(stats['steady_samples'], 6)
        self.assertEqual(stats['process_means'], [2, 4, 6])
        self.assertEqual(stats['process_std'], 2)

    def test_sync_speedup_uses_same_lifecycle_scope(self):
        rows = []
        for backend in ('eager', 'graph'):
            for mode, source, ready, verified in [('fresh', 'gpu-fp32', 100, 102),
                ('hot', 'gpu-fp32', 4, 6), ('hot', 'cpu-bf16', 8, 10)]:
                rows.append({'backend': backend, 'mode': mode, 'source': source,
                    'preflight_seconds': .6, 'commit_seconds': .3, 'invalidate_seconds': .1,
                    'sync_seconds': 1, 'recovery_to_ready_seconds': ready,
                    'recovery_including_verified_rollout_seconds': verified})
        stats = sync_statistics(rows)['eager']
        self.assertEqual(stats['ready_speedup'], 25)
        self.assertEqual(stats['verified_generation_speedup'], 17)

    def test_followups_match_everything_except_seed_anchor_and_update_count(self):
        runs = configurations('/persistent/model')
        self.assertEqual(len(runs), 8)
        for config in runs.values():
            self.assertEqual(config['learning_rate'], 1e-6)
            self.assertEqual(config['iterations'], 120)
            self.assertEqual(config['prompts_per_iteration'] * config['grpo']['group_size'], 128)
            self.assertIn('GPU_PENDING', config['validation_status'])
        for seed in (0, 1, 2):
            a, b = (runs[f'anchor-{anchor}-seed{seed}'] for anchor in ('rollout', 'recomputed'))
            self.assertEqual(a['grpo']['update_steps'], 2)
            self.assertEqual(b['grpo']['update_steps'], 2)
            self.assertEqual({k: v for k, v in a.items() if k != 'grpo'}, {k: v for k, v in b.items() if k != 'grpo'})
            self.assertEqual({k: v for k, v in a['grpo'].items() if k != 'anchor'}, {k: v for k, v in b['grpo'].items() if k != 'anchor'})


if __name__ == '__main__':
    unittest.main()
