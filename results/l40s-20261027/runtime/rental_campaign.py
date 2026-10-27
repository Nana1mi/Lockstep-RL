"""Server-side serial campaign: clean-commit preflight, then full experiment plan."""
import fcntl
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/root/gpufree-data/lockstep')
REPO = ROOT / 'repo-ce9e354'
OUT = ROOT / 'results/campaign-ce9e354-01'
EXPECTED = 'ce9e3542b3f31475ffa595e3f09b80e3f9190ed0'
PYTHON = str(ROOT / 'envs/nano_rl/bin/python')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(state):
    state['updated_unix'] = time.time()
    temp = OUT / 'manifest.pending'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(OUT / 'manifest.json')


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    lock = (ROOT / 'ops/gpu-campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Campaign already exists; do not overwrite or start duplicate jobs')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s UTC %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / 'console.log')])
    logging.Formatter.converter = time.gmtime
    sys.path.insert(0, str(REPO))
    from lockstep_rl.process_runner import run_task
    import importlib.util
    spec = importlib.util.spec_from_file_location('source_guard', ROOT / 'ce9-launch-audit/launch_ce9e354.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    source = guard.identity(REPO)
    state = {'status': 'starting', 'source_commit': EXPECTED, 'source': source,
        'started_unix': time.time(), 'phases': [], 'active_run_id': None,
        'rental_deadline': None, 'policy': 'Run all planned experiments serially; per-task timeouts retained; no fixed ten-hour cutoff; never renew or stop platform instance'}
    save(state)
    env = dict(os.environ)
    for name in ('LOCKSTEP_TASK_DEADLINE', 'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env.pop(name, None)
    env.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(ROOT / 'cache/huggingface'), PIP_CACHE_DIR=str(ROOT / 'cache/pip'),
        TORCH_HOME=str(ROOT / 'cache/torch'), TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/torchinductor'), TMPDIR=str(ROOT / 'tmp'))
    power = 'L40S observed power limit350W; platform default clocks, no clock changes'
    stop = threading.Event()
    def progress():
        while not stop.wait(30):
            run = state.get('active_run_id')
            if run:
                m = read(ROOT / 'results' / run / 'manifest.json')
                done = m.get('completed_jobs', [])
                jobs = m.get('jobs', [])
                current = jobs[len(done)]['name'] if len(done) < len(jobs) else m.get('status', 'initializing')
                detail = {}
                for child in ('baseline', 'diagnostics', 'weight-sync', 'performance'):
                    c = read(ROOT / 'results' / run / child / 'manifest.json')
                    if c:
                        detail[child] = {'status': c.get('status'), 'finished_workers': len(c.get('jobs', []))}
                logging.info('run=%s stages=%d/%d current=%s detail=%s', run, len(done), len(jobs), current, detail)
    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    try:
        for run_id, stages in [('preflight-ce9e354-01', ['environment', 'baseline']), ('full-ce9e354-01', None)]:
            guard.identity(REPO)
            state.update(status='running', active_run_id=run_id)
            save(state)
            cmd = [PYTHON, str(ROOT / 'ce9-launch-audit/launch_ce9e354.py'), '--repo', str(REPO), '--',
                '--persistent-root', str(ROOT), '--python', PYTHON, '--run-id', run_id,
                '--power-mode', power, '--allow-gpu']
            if stages:
                cmd.extend(['--stages', *stages])
            # Aggregate bound is the sum of individual caps, not a rental/time-of-day deadline.
            plan = read(ROOT / 'ce9-launch-audit/full-dry-run.json')
            cap = 5400 if stages else sum(j['timeout_seconds'] for j in plan['jobs']) + 1200
            logging.info('START %s source=%s task-cap-sum=%s', run_id, EXPECTED, cap)
            with (OUT / f'{run_id}.log').open('w') as log:
                result = run_task(cmd, timeout_seconds=cap, cwd=REPO, env=env, stdout=log)
            phase = {'run_id': run_id, 'command': cmd, **result.record()}
            state['phases'].append(phase)
            save(state)
            logging.info('FINISH %s returncode=%s elapsed=%.1fs', run_id, result.returncode, result.wall_seconds)
            manifest = read(ROOT / 'results' / run_id / 'manifest.json')
            if manifest.get('source_commit') != EXPECTED:
                raise RuntimeError('Run manifest failed source commit gate')
            guard.identity(REPO)
            if stages:
                if result.returncode or manifest.get('failed_jobs'):
                    raise RuntimeError('Clean-commit preflight failed; full run not started')
                observed = manifest['completed_jobs'] + read(ROOT / 'results' / run_id / 'baseline/manifest.json')['jobs']
                risky = [x['name'] for x in observed if x['wall_seconds'] > x['timeout_seconds'] / 2]
                state['preflight_timing_audit'] = {'checked': len(observed), 'over_half_cap': risky}
                save(state)
                if risky:
                    raise RuntimeError('Fresh preflight exceeds half cap; re-budget before full run: ' + str(risky))
            else:
                state['status'] = 'completed' if result.returncode == 0 else 'completed_with_failures'
                state['failed_jobs'] = manifest.get('failed_jobs', [])
        state['finished_unix'] = time.time()
        state['active_run_id'] = None
        save(state)
        logging.info('CAMPAIGN %s failures=%s', state['status'], state.get('failed_jobs', []))
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save(state)
        logging.exception('CAMPAIGN STOPPED')
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
