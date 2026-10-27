"""Read-only campaign status; no CUDA initialization or model loading."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/root/gpufree-data/lockstep')
CAMPAIGN = ROOT / 'results/campaign-ce9e354-01'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, limit=6000):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - limit))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return {'returncode': p.returncode, 'stdout': p.stdout[-16000:], 'stderr': p.stderr[-1000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def snapshot():
    state = read(CAMPAIGN / 'manifest.json')
    disk = shutil.disk_usage(ROOT)
    mem = {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    cg = Path('/sys/fs/cgroup')
    limit = (cg / 'memory.max').read_text().strip()
    current = int((cg / 'memory.current').read_text())
    quota, period = (cg / 'cpu.max').read_text().split()
    cpu_stat = {line.split()[0]: int(line.split()[1]) for line in (cg / 'cpu.stat').read_text().splitlines()}
    answer = {'unix_time': time.time(), 'campaign': state, 'host_load_average': os.getloadavg(),
        'host_ram': {k: mem[k] for k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')},
        'container_memory': {'limit_bytes': int(limit) if limit != 'max' else None,
                             'usage_including_cache_bytes': current},
        'container_cpu_quota': float(quota) / float(period) if quota != 'max' else None,
        'container_cpu_stat': cpu_stat,
        'archive': read(CAMPAIGN / 'archive.json'),
        'source_head': command(['git', '-C', str(ROOT / 'repo-ce9e354'), 'rev-parse', 'HEAD']),
        'source_tracked_status': command(['git', '-C', str(ROOT / 'repo-ce9e354'), 'status', '--porcelain', '--untracked-files=no']),
        'disk': {'total_bytes': disk.total, 'free_bytes': disk.free},
        'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit', '--format=csv']),
        'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
        'runs': {}, 'training': {}, 'campaign_log_tail': tail(CAMPAIGN / 'console.log')}
    for run_id in ('preflight-ce9e354-01', 'full-ce9e354-01'):
        folder = ROOT / 'results' / run_id
        m = read(folder / 'manifest.json')
        if not m:
            continue
        done, jobs = m.get('completed_jobs', []), m.get('jobs', [])
        active = jobs[len(done)]['name'] if len(done) < len(jobs) else None
        row = {'source_commit': m.get('source_commit'), 'status': m.get('status'),
            'finished_stages': len(done), 'total_stages': len(jobs), 'active_job': active,
            'failed_jobs': m.get('failed_jobs', []), 'timeout_overrides': m.get('timeout_overrides', []), 'children': {}}
        for name, total in [('baseline', 24), ('diagnostics', 14), ('weight-sync', 25), ('performance', 96)]:
            child = read(folder / name / 'manifest.json')
            if child:
                records = child.get('jobs', [])
                row['children'][name] = {'status': child.get('status'), 'finished_workers': len(records),
                    'planned_workers': total, 'failed_workers': [x['name'] for x in records if x.get('returncode')],
                    'last_worker': records[-1]['name'] if records else None}
        if active:
            row['active_console_tail'] = tail(folder / (active + '.console.log'))
        answer['runs'][run_id] = row
        for name in ('two-updates', 'smoke-rollout', 'smoke-recomputed', 'train-demo-simple', 'train-demo-simple-lr5e-6', 'gsm8k-smoke'):
            path = folder / name / 'metrics.csv'
            if path.exists():
                with path.open() as stream:
                    header = stream.readline()
                lines = tail(path, 16000).splitlines()
                if len(lines) > 1:
                    rows = list(csv.DictReader(io.StringIO(header + '\n'.join(lines[-1:]))))
                    if rows:
                        answer['training'][run_id + '/' + name] = rows[-1]
    return answer


def display(s):
    c = s['campaign']
    print(datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S HKT'))
    print('Campaign:', c.get('status', 'not started'), '| active:', c.get('active_run_id'), '| commit:', c.get('source_commit'))
    m = s['container_memory']
    print('Container CPU quota:', s['container_cpu_quota'], '| RAM usage incl cache: %.2f / %.2f GiB' % (m['usage_including_cache_bytes'] / 2**30, (m['limit_bytes'] or 0) / 2**30), '| data disk free: %.2f GiB' % (s['disk']['free_bytes'] / 2**30))
    if 'sampled_cpu_percent_of_quota' in s:
        print('CPU usage within allocation: %.1f%%' % s['sampled_cpu_percent_of_quota'])
    print(s['gpu'].get('stdout', s['gpu'].get('error', 'GPU unavailable')).strip())
    for run_id, row in s['runs'].items():
        print(f"{run_id}: {row['status']} stages {row['finished_stages']}/{row['total_stages']}; active={row['active_job']}; failures={row['failed_jobs']}")
        for name, v in row['children'].items():
            print(f"  {name}: {v['status']} workers {v['finished_workers']}/{v['planned_workers']}; failed={v['failed_workers']}; last={v['last_worker']}")
    for name, v in s['training'].items():
        print(f"  {name}: iteration={v.get('iteration')}, reward={v.get('mean_reward')}, loss={v.get('loss')}, policy={v.get('policy_version')}, updated={v.get('updated')}")
    if c.get('error'):
        print('ERROR:', c['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json', action='store_true')
    p.add_argument('--watch', type=float, default=0)
    args = p.parse_args()
    previous = None
    while True:
        s = snapshot()
        if previous is not None and s['container_cpu_quota']:
            elapsed = s['unix_time'] - previous['unix_time']
            used = s['container_cpu_stat']['usage_usec'] - previous['container_cpu_stat']['usage_usec']
            s['sampled_cpu_percent_of_quota'] = 100 * used / 1e6 / elapsed / s['container_cpu_quota']
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            display(s)
        if not args.watch:
            break
        previous = s
        time.sleep(max(args.watch, 2))


if __name__ == '__main__':
    main()
