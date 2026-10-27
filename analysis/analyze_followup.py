"""Recompute follow-up seed curves and per-update clipping on CPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'results/followup-l40s-20261027'

def read_csv(path):
    with path.open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))

def analyze(data):
    summary = json.loads((data / 'summary.json').read_text())
    groups = {'replication': [], 'anchor-rollout': [], 'anchor-recomputed': []}
    inputs = {}
    for job in summary['jobs']:
        folder = data / 'training' / job['name']
        for name in ('config.json', 'metrics.csv', 'steps.csv', 'validation-initial-summary.json', 'summary.json'):
            path = folder / name
            inputs[path.relative_to(data).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = json.loads((folder / 'config.json').read_text())
        metrics = read_csv(folder / 'metrics.csv')
        steps = read_csv(folder / 'steps.csv')
        initial = json.loads((folder / 'validation-initial-summary.json').read_text())
        if len(metrics) != 120 or [int(r['iteration']) for r in metrics] != list(range(120)):
            raise ValueError('Missing or duplicate training iterations')
        if len({(int(r['iteration']), int(r['index'])) for r in steps}) != len(steps):
            raise ValueError('Duplicate per-update rows')
        if any(not math.isfinite(float(r['clip_active_fraction'])) or not 0 <= float(r['clip_active_fraction']) <= 1 for r in steps):
            raise ValueError('Invalid clipping fraction')
        active = sum(float(r['clip_active_fraction']) > 0 for r in steps)
        if active != job['steps_with_active_clipping'] or len(steps) != job['recorded_steps']:
            raise ValueError('Derived steps disagree with original JSONL summary')
        curve = [[0, initial['accuracy']]] + [[int(r['iteration']) + 1, float(r['validation_accuracy'])]
            for r in metrics if r['validation_accuracy']]
        if [x for x, _ in curve] != list(range(0, 121, 10)):
            raise ValueError('Evaluation rounds differ')
        if curve[-1][1] != job['final_accuracy']:
            raise ValueError('Final validation differs')
        group = 'replication' if job['name'].startswith('replication-') else 'anchor-' + config['grpo']['anchor']
        groups[group].append({'seed': config['seed'], 'curve': curve, 'steps': len(steps), 'active_clipping_updates': active})
    for name, group in groups.items():
        group.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in group] != [0, 1, 2]:
            raise ValueError('Each group must have exactly seeds0/1/2')
        curve = []
        for index in range(13):
            values = [r['curve'][index][1] for r in group]
            curve.append({'round': index * 10, 'mean': statistics.fmean(values), 'sample_sd': statistics.stdev(values)})
        groups[name] = {'seeds': group, 'curve': curve, 'active_clipping_updates': sum(r['active_clipping_updates'] for r in group),
            'recorded_updates': sum(r['steps'] for r in group)}
        final = summary['groups'][name]
        if abs(curve[-1]['mean'] - final['mean_final_accuracy']) > 1e-12 or abs(curve[-1]['sample_sd'] - final['sample_seed_sd']) > 1e-12:
            raise ValueError('Seed aggregation differs from original summary')
    return {'source_commit': summary['source_commit'], 'input_sha256': inputs, 'groups': groups,
        'scope': 'Three seeds on fixed128 structured validation tasks, temperature1.0. Shading is sample SD, not CI, and includes evaluation sampling randomness. No anchor superiority claim.'}

def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    left = result['groups']['replication']
    colors = ['#2864b5', '#209b82', '#b76e1d']
    for row, color in zip(left['seeds'], colors):
        axes[0].plot([x[0] for x in row['curve']], [100*x[1] for x in row['curve']], '-o', ms=3,
            alpha=.65, color=color, lw=1, label='seed ' + str(row['seed']))
    def aggregate(ax, group, color, label):
        curve = group['curve']
        rounds = [r['round'] for r in curve]
        means = [100*r['mean'] for r in curve]
        low = [100*(r['mean'] - r['sample_sd']) for r in curve]
        high = [100*(r['mean'] + r['sample_sd']) for r in curve]
        ax.plot(rounds, means, color=color, lw=2, label=label)
        ax.fill_between(rounds, low, high, color=color, alpha=.15)
    aggregate(axes[0], left, '#333333', 'mean +/- sample SD')
    aggregate(axes[1], result['groups']['anchor-rollout'], colors[0], 'rollout')
    aggregate(axes[1], result['groups']['anchor-recomputed'], colors[1], 'recomputed')
    for ax, title in zip(axes, ('Recomputed anchor; one update per batch', 'Matched anchors; two updates per batch')):
        ax.set(xlabel='Training round', ylabel='Sampled validation accuracy (%)', title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Three seeds; shading is SD, not CI (includes evaluation sampling)', fontsize=11)
    fig.savefig(output / 'validation-curves.png', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(args.data)
    (args.output / 'recomputed.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    if not args.no_plots:
        plot(result, args.output)
    print(json.dumps({'groups': len(result['groups']), 'input_hashes': len(result['input_sha256']),
        'updates': sum(g['recorded_updates'] for g in result['groups'].values()), 'scope': 'CPU only'}))

if __name__ == '__main__':
    main()
