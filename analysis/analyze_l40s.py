"""Recompute published L40S measurements on CPU; never imports torch or starts jobs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/l40s-20261027/full'


def quantile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError('nonempty data and probability in [0,1] required')
    values = sorted(values)
    position = (len(values) - 1) * probability
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def csv_rows(path):
    with path.open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def token_key(row):
    return row['request_id'], int(row['response_index']), int(row['token_id'])


def token_statistics(rows):
    if not rows or len({(row['request_id'], int(row['response_index'])) for row in rows}) != len(rows):
        raise ValueError('nonempty unique aligned token rows required')
    if any(row['segment'] != 'primary' for row in rows):
        raise ValueError('primary-only input required')
    delta = [float(row['delta_logp']) for row in rows]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError('finite scores required')
    errors = [abs(x) for x in delta]
    maxima = defaultdict(float)
    for row, error in zip(rows, errors):
        maxima[row['request_id']] = max(maxima[row['request_id']], error)
    return {'tokens': len(rows), 'trajectories': len(maxima), 'mean_absolute_error': statistics.fmean(errors),
        'p99_absolute_error': quantile(errors, .99), 'max_absolute_error': max(errors),
        'fraction_above_0_1': sum(x > .1 for x in errors) / len(errors),
        'trajectories_above_0_1': sum(x > .1 for x in maxima.values()),
        'ratio_outside_0_8_1_2': sum(not .8 <= math.exp(x) <= 1.2 for x in delta) / len(delta),
        'per_trajectory_maximum': dict(maxima)}


def paired_binary(initial, final):
    def index(rows):
        mapping = {row['id']: float(row['reward']) for row in rows}
        if len(mapping) != len(rows) or any(x not in (0, 1) for x in mapping.values()):
            raise ValueError('unique IDs with binary outcomes required')
        return mapping
    left, right = index(initial), index(final)
    if not left or left.keys() != right.keys():
        raise ValueError('identical nonempty paired task IDs required')
    differences = [right[key] - left[key] for key in left]
    gained, lost = differences.count(1), differences.count(-1)
    n, discordant = len(left), gained + lost
    se = statistics.stdev(differences) / math.sqrt(n) if n > 1 else None
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / 2 ** discordant) if discordant else 1.
    return {'n': n, 'gained': gained, 'lost': lost, 'difference': statistics.fmean(differences),
        'paired_standard_error': se, 'difference_over_se': statistics.fmean(differences) / se if se else None,
        'exact_mcnemar_two_sided_p': p,
        'scope': 'Conditional paired validation outcomes; structured tasks and one training seed. Not training-seed uncertainty or a robust-learning significance test.'}


def sync_statistics(rows):
    result = {}
    for backend in ('eager', 'graph'):
        fresh = [r for r in rows if r['backend'] == backend and r['mode'] == 'fresh']
        hot = [r for r in rows if r['backend'] == backend and r['mode'] == 'hot' and r['source'] == 'gpu-fp32']
        cpu = [r for r in rows if r['backend'] == backend and r['mode'] == 'hot' and r['source'] == 'cpu-bf16']
        mean = lambda records, key: statistics.fmean(float(r[key]) for r in records)
        values = {key: mean(hot, key) for key in ('preflight_seconds', 'commit_seconds', 'invalidate_seconds',
            'sync_seconds', 'recovery_to_ready_seconds', 'recovery_including_verified_rollout_seconds')}
        values.update(fresh_ready_seconds=mean(fresh, 'recovery_to_ready_seconds'),
            fresh_verified_seconds=mean(fresh, 'recovery_including_verified_rollout_seconds'),
            cpu_bf16_sync_seconds=mean(cpu, 'sync_seconds'), repeats=len(hot))
        values['ready_speedup'] = values['fresh_ready_seconds'] / values['recovery_to_ready_seconds']
        values['verified_generation_speedup'] = values['fresh_verified_seconds'] / values['recovery_including_verified_rollout_seconds']
        result[backend] = values
    return result


def process_statistics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['process_repeat'])].append(float(row['output_tokens_per_second']))
    values = [statistics.fmean(groups[key]) for key in sorted(groups)]
    return {'mean': statistics.fmean(values), 'process_means': values,
        'process_std': statistics.stdev(values) if len(values) > 1 else 0., 'independent_processes': len(values),
        'steady_samples': sum(map(len, groups.values()))}


def analyze(raw_archive=None):
    comparisons = {}
    source_rows = {}
    paths = [*sorted(DATA.rglob('*.csv')), ROOT / 'results/l40s-20261027/training-summary.json']
    inputs = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for path in sorted((DATA / 'diagnostics').glob('*.csv')):
        if path.stem.endswith('-post-eos'):
            continue
        source_rows[path.stem] = csv_rows(path)
        comparisons[path.stem] = token_statistics(source_rows[path.stem])
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    batch = {token_key(row): float(row['reference_logprob']) for row in source_rows['path-batch']}
    padding = {token_key(row): float(row['reference_logprob']) for row in source_rows['path-graph_padding']}
    if batch.keys() != padding.keys():
        raise ValueError('batch and padding token IDs/positions differ')
    equality = {'tokens': len(batch), 'different_scores': sum(batch[key] != padding[key] for key in batch),
        'scope': 'Two tested groupings only. Backbone graph bucket4 vs eager batch4; graph head receives actual3 rows after slicing.'}
    bins = {}
    for label, predicate in [('rollout_logp_below_minus1', lambda x: x < -1),
                              ('rollout_logp_at_least_minus0_1', lambda x: x >= -.1)]:
        values = [abs(float(r['delta_logp'])) for r in source_rows['nano-vs-hf-fp32'] if predicate(float(r['rollout_logprob']))]
        bins[label] = {'tokens': len(values), 'mean_absolute_error': statistics.fmean(values)}
    performance = []
    rows = csv_rows(DATA / 'performance/measurements.csv')
    for backend in ('eager', 'graph'):
        for case in ('short-single', 'short-multi', 'long-single', 'long-multi'):
            stats = {variant: process_statistics([r for r in rows if r['backend'] == backend and r['case'] == case and r['source_variant'] == variant])
                for variant in ('baseline', 'current', 'current-fp32-head-live', 'current-fp32-head-cached')}
            mean = lambda variant: stats[variant]['mean']
            performance.append({'backend': backend, 'case': case, 'groups': stats,
                'extension_change_percent': 100 * (mean('current') / mean('baseline') - 1),
                'live_head_change_percent': 100 * (mean('current-fp32-head-live') / mean('current') - 1),
                'cached_head_change_percent': 100 * (mean('current-fp32-head-cached') / mean('current') - 1)})
    training = {}
    for name in ('train-demo-simple', 'train-demo-simple-lr5e-6'):
        metrics = csv_rows(DATA / name / 'metrics.csv')
        initial = json.loads((ROOT / 'results/l40s-20261027/training-summary.json').read_text())[name]['initial_validation']
        training[name] = {'iterations': len(metrics), 'initial_accuracy': initial['accuracy'],
            'final_accuracy': float(metrics[-1]['validation_accuracy']),
            'minimum_validation': min((float(r['validation_accuracy']), int(r['iteration'])) for r in metrics if r['validation_accuracy']),
            'maximum_clip_active_fraction': max(float(r['clip_active_fraction']) for r in metrics if r['clip_active_fraction']),
            'updated_iterations': sum(r['updated'] == 'True' for r in metrics),
            'evaluation_curve': [[0, initial['accuracy']]] + [[int(r['iteration']) + 1, float(r['validation_accuracy'])] for r in metrics if r['validation_accuracy']]}
    raw = {'available': False}
    if raw_archive:
        raw = {'available': True, 'archive_sha256': hashlib.sha256(raw_archive.read_bytes()).hexdigest(), 'negative_controls': [], 'decode_execution_shapes': {}, 'smoke_steps': {}}
        with zipfile.ZipFile(raw_archive) as archive:
            for name in training:
                prefix = f'results/full-ce9e354-01/{name}/'
                initial = json.loads(archive.read(prefix + 'validation-initial.json'))
                final = json.loads(archive.read(prefix + 'metrics.jsonl').splitlines()[-1])['validation']
                training[name]['paired_validation'] = paired_binary(initial['rows'], final['rows'])
            for name in archive.namelist():
                if '/weight-sync/' in name and '-negative-' in name and name.endswith('/vs-fresh.json'):
                    report = json.loads(archive.read(name))
                    raw['negative_controls'].append({'name': name.split('/')[-2], 'detected': report['negative_control_detected'],
                        'max_error': report['absolute_logprob_error']['max']})
            for case in ('eager', 'graph', 'batch', 'graph_padding'):
                trace = json.loads(archive.read(f'results/full-ce9e354-01/diagnostics/{case}/trace.json'))
                counts = Counter((e.get('actual_batch'), e.get('graph_batch'), e.get('graph_replay'))
                    for e in trace if e.get('event') == 'execution' and not e.get('is_prefill'))
                raw['decode_execution_shapes'][case] = [{'actual_batch': a, 'graph_batch': g, 'graph_replay': replay, 'events': count}
                    for (a, g, replay), count in counts.items()]
            for case in ('smoke-rollout', 'smoke-recomputed', 'two-updates'):
                records = [json.loads(x) for x in archive.read(f'results/full-ce9e354-01/{case}/metrics.jsonl').splitlines()]
                raw['smoke_steps'][case] = [{'iteration': row['iteration'], **{k: step[k] for k in
                    ('index', 'ratio_min', 'ratio_max', 'ratio_outside_fraction', 'clip_active_fraction')}}
                    for row in records for step in row['steps']]
    return {'experimental_source_commit': 'ce9e3542b3f31475ffa595e3f09b80e3f9190ed0',
        'scope': 'One Qwen3-0.6B model, one L40S, fixed32 trajectories; finite-sample observations, not universal bounds.',
        'input_sha256': inputs, 'comparisons': comparisons, 'batch_padding_equality': equality, 'probability_bins': bins,
        'synchronization': sync_statistics(csv_rows(DATA / 'weight-sync/timings.csv')),
        'performance': performance, 'training': training, 'raw_archive_checks': raw}


def save_figure(figure, path):
    figure.savefig(path, dpi=180)
    if path.suffix == '.svg':
        # Matplotlib emits trailing spaces inside multiline path attributes.
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n', encoding='utf8')


def figures(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'lockstep-l40s'})
    palette = ['#2864b5', '#209b82', '#b76e1d', '#9c488a']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout='constrained')
    for i, (key, label) in enumerate([('nano-vs-hf-fp32', 'nano BF16 vs HF FP32'),
        ('nano-fp32-head-vs-hf-fp32', 'nano FP32 head vs HF FP32'),
        ('hf-bf16-vs-hf-fp32', 'HF BF16 vs HF FP32'), ('nano-vs-hf-bf16', 'nano BF16 vs HF BF16')]):
        values = sorted(max(abs(float(r['delta_logp'])), 1e-7) for r in csv_rows(DATA / 'diagnostics' / (key + '.csv')))
        axes[0].step(values, [(j + 1) / len(values) for j in range(len(values))], where='post', label=label, color=palette[i])
    axes[0].set(xscale='log', xlabel='Absolute logprob error (zero displayed at 1e-7)', ylabel='Empirical cumulative fraction', title='3,361 tokens through first EOS; 32 trajectories')
    axes[0].axvline(.1, color='#777', ls=':', lw=1)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(alpha=.15)
    keys = [('fresh_verified_seconds', 'Reload + verified output'),
            ('recovery_including_verified_rollout_seconds', 'Hot update + verified output'), ('sync_seconds', 'Sync API phases only')]
    for i, (key, label) in enumerate(keys):
        axes[1].bar([j + (i - 1) * .24 for j in range(2)],
            [result['synchronization'][b][key] for b in ('eager', 'graph')], width=.23, label=label, color=palette[i])
    axes[1].set(yscale='log', xticks=[0, 1], xticklabels=['eager', 'graph'], ylabel='Seconds; different scopes explicitly separated', title='Weight lifecycle; three repeats per configuration')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(axis='y', alpha=.15)
    for extension in ('png', 'svg'):
        save_figure(fig, output / ('diagnostics-sync.' + extension))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for i, (name, label) in enumerate([('train-demo-simple', 'lr=1e-6'), ('train-demo-simple-lr5e-6', 'lr=5e-6')]):
        curve = result['training'][name]['evaluation_curve']
        axes[0].plot([x[0] for x in curve], [100 * x[1] for x in curve], '-o', ms=3.5, color=palette[i], label=label)
    axes[0].set(xlabel='Training round (iteration + 1)', ylabel='Validation accuracy (%)', ylim=(35, 100), title='128 synthetic multiplication tasks; seed0 only')
    axes[0].legend()
    axes[0].grid(alpha=.2)
    rows = result['performance']
    for i, (variant, label) in enumerate([('current', 'BF16 full rollout contract'), ('current-fp32-head-live', 'FP32 head live'), ('current-fp32-head-cached', 'FP32 head cached')]):
        heights = [row['groups'][variant]['mean'] / row['groups']['baseline']['mean'] for row in rows]
        deviations = [row['groups'][variant]['process_std'] / row['groups']['baseline']['mean'] for row in rows]
        axes[1].bar([j + (i - 1) * .25 for j in range(len(rows))], heights, yerr=deviations, width=.23, color=palette[i], label=label, capsize=2)
    aliases = {'short-single': 'S1', 'short-multi': 'S4', 'long-single': 'L1', 'long-multi': 'L4'}
    axes[1].set(xticks=range(8), xticklabels=[r['backend'] + '\n' + aliases[r['case']] for r in rows],
        ylabel='Steady throughput / same-case upstream mean', ylim=(0, 1.15), title='Three process repeats; bars show mean +/- process SD')
    axes[1].axhline(1, color='#777', ls=':', lw=1)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].legend(fontsize=7.5, loc='lower right')
    for extension in ('png', 'svg'):
        save_figure(fig, output / ('training-performance.' + extension))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-archive', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/l40s-20261027')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(args.raw_archive)
    (args.output / 'recomputed.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    if not args.no_plots:
        figures(result, args.output)
    print(json.dumps({'output': str(args.output), 'raw_archive_checks': result['raw_archive_checks']['available'],
        'tokens': result['comparisons']['nano-vs-hf-fp32']['tokens'], 'comparisons': len(result['comparisons'])}))


if __name__ == '__main__':
    main()
