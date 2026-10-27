"""Serial follow-up execution outside the immutable ce9e354 checkout."""
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path('/root/gpufree-data/lockstep')
REPO = ROOT / 'repo-ce9e354'
OUT = ROOT / 'results/followup-ce9e354-01'
EXPECTED = 'ce9e3542b3f31475ffa595e3f09b80e3f9190ed0'
PYTHON = str(ROOT / 'envs/nano_rl/bin/python')

def read(path):
    return json.loads(path.read_text())

def save(state):
    state['updated_unix'] = time.time()
    pending = OUT / 'manifest.pending'
    pending.write_text(json.dumps(state, indent=2) + '\n')
    pending.replace(OUT / 'manifest.json')

def command(args):
    return subprocess.check_output(args, cwd=REPO, text=True, timeout=30).strip()

def guard():
    assert command(['git', 'rev-parse', 'HEAD']) == EXPECTED, 'Source commit changed'
    assert not command(['git', 'status', '--porcelain', '--untracked-files=no']), 'Source dirty'
    from scripts.run_rental import gpu_inventory
    from lockstep_rl.rental import validate_preflight
    validate_preflight(read(OUT / 'preflight.json'), ROOT, gpu_inventory('0'), read(REPO / 'configs/rental.json'))
    assert shutil.disk_usage(ROOT).free > 5 * 1024**3, 'Less than 5GiB free on data disk'

def preflight_gate():
    pre = ROOT / 'results/preflight-followup-ce9e354-01'
    record = read(pre / 'manifest.json')
    done = record.get('completed_jobs', [])
    assert record.get('status') == 'executed_review_raw_results'
    assert not record.get('failed_jobs') and len(done) == 2
    assert {j['name'] for j in done} == {'environment', 'attention'}
    assert all(j['returncode'] == 0 and not j['timed_out'] for j in done)
    assert read(pre / 'environment.json')['git']['stdout'].strip() == EXPECTED

def export(state):
    target = ROOT / 'exports/followup-ce9e354-01-results.zip'
    assert not target.exists(), 'Preserve existing export'
    excluded = {'__pycache__', '.git', 'checkpoint-final', 'checkpoints'}
    entries = {}
    with zipfile.ZipFile(target.with_suffix('.pending'), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder in (OUT, ROOT / 'results/preflight-followup-ce9e354-01'):
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.is_symlink() or set(path.parts) & excluded:
                    continue
                if path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.pyc', '.zip'} or path.name in {'train-tasks.json', 'validation-tasks.json'}:
                    continue
                name = str(path.relative_to(ROOT))
                archive.write(path, name)
                entries[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for name in ('followup_campaign.py', 'followup_monitor.py'):
            path = ROOT / 'ops' / name
            archive.write(path, 'ops/' + name)
        archive.writestr('SOURCE_COMMIT.txt', EXPECTED + '\n')
        archive.writestr('MANIFEST.json', json.dumps(entries, indent=2) + '\n')
    with zipfile.ZipFile(target.with_suffix('.pending')) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
    target.with_suffix('.pending').replace(target)
    report = {'status': 'exported', 'zip': str(target), 'bytes': target.stat().st_size,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'files': len(entries), 'created_unix': time.time()}
    (OUT / 'archive.json').write_text(json.dumps(report, indent=2) + '\n')

def main():
    sys.path.insert(0, str(REPO))
    from lockstep_rl.process_runner import run_task
    lock = (ROOT / 'ops/gpu-campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'manifest.json').exists(), 'Campaign already exists; never overwrite'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / 'console.log')])
    plan = read(OUT / 'plan.json')
    state = {'status': 'starting', 'source_commit': EXPECTED, 'plan': plan,
        'started_unix': time.time(), 'completed_jobs': [], 'failed_jobs': [], 'active_job': None,
        'preflight': read(OUT / 'preflight.json'), 'rental_deadline': None,
        'input_precheck': read(OUT / 'input-precheck.json'),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(state)
    stop = threading.Event()
    def progress():
        while not stop.wait(30):
            name = state.get('active_job')
            latest = {}
            if name:
                path = OUT / name / 'metrics.csv'
                if path.exists():
                    import csv
                    with path.open() as f:
                        rows = list(csv.DictReader(f))
                    latest = rows[-1] if rows else {}
            logging.info('progress completed=%s/%s active=%s latest=%s',
                len(state['completed_jobs']), len(plan['jobs']), name,
                {k: latest.get(k) for k in ('iteration', 'mean_reward', 'clip_active_fraction', 'validation_accuracy')})
    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    env = dict(os.environ)
    for key in ('LOCKSTEP_TASK_DEADLINE', 'LOCKSTEP_NCCL_PORT', 'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        LOCKSTEP_RENTAL_REGION='安徽区', HF_HOME=str(ROOT / 'cache/huggingface'), TORCH_HOME=str(ROOT / 'cache/torch'),
        TRITON_CACHE_DIR=str(ROOT / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/torchinductor'), TMPDIR=str(ROOT / 'tmp'))
    try:
        preflight_gate()
        for job in plan['jobs']:
            guard()
            name = job['name']
            config = OUT / 'configs' / (name + '.json')
            assert hashlib.sha256(config.read_bytes()).hexdigest() == job['config_sha256'], 'Config changed'
            state.update(status='running', active_job=name)
            save(state)
            cmd = [PYTHON, 'scripts/train_grpo.py', '--config', str(config), '--output-dir', str(OUT / name), '--allow-gpu']
            logging.info('START %s timeout=%ss', name, job['timeout_seconds'])
            with (OUT / (name + '.log')).open('w') as log:
                task = run_task(cmd, timeout_seconds=job['timeout_seconds'], cwd=REPO, env=env, stdout=log, nccl=True)
            record = {'name': name, 'command': cmd, **task.record()}
            state['completed_jobs'].append(record)
            if task.returncode:
                state['failed_jobs'].append(name)
            save(state)
            logging.info('FINISH %s returncode=%s elapsed=%.1fs', name, task.returncode, task.wall_seconds)
            guard()
        state.update(status='completed_with_failures' if state['failed_jobs'] else 'completed', active_job=None, finished_unix=time.time())
        save(state)
        logging.info('CAMPAIGN %s failed=%s', state['status'], state['failed_jobs'])
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', active_job=None, finished_unix=time.time())
        save(state)
        logging.exception('CAMPAIGN STOPPED')
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        try:
            export(state)
        except Exception as exc:
            (OUT / 'archive.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}))
            logging.exception('EXPORT FAILED')

if __name__ == '__main__':
    main()