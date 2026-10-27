"""Create a small campaign ZIP after completion; excludes weights/envs/checkpoints."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path('/root/gpufree-data/lockstep')
OUT = ROOT / 'results/campaign-ce9e354-01'


def main():
    while True:
        try:
            state = json.loads((OUT / 'manifest.json').read_text())
        except (OSError, ValueError):
            state = {}
        if state.get('status') in ('completed', 'completed_with_failures', 'failed'):
            break
        time.sleep(30)
    destination = ROOT / 'exports/campaign-ce9e354-01-results.zip'
    destination.parent.mkdir(exist_ok=True)
    excluded = {'upstream-source', '__pycache__', '.git', 'checkpoint-final', 'checkpoints'}
    entries = {}
    for name in ('campaign-ce9e354-01', 'preflight-ce9e354-01', 'full-ce9e354-01'):
        for file in (ROOT / 'results' / name).rglob('*'):
            rel = file.relative_to(ROOT)
            if (file.is_file() and not set(rel.parts) & excluded
                    and file.suffix not in ('.safetensors', '.pt', '.pth', '.bin', '.whl', '.pyc', '.zip')
                    and not file.is_symlink()):
                entries[str(rel)] = file
    for name in ('rental-preflight.json', 'models/Qwen3-0.6B/_lockstep_model.json',
                 'data/gsm8k/manifest.json', 'envs/nano_rl/environment-freeze.txt',
                 'ce9-launch-audit/launch_ce9e354.py', 'ops/rental_campaign.py',
                 'ops/monitor_status.py', 'ops/export_results.py', 'ops/REMOTE_COMMANDS.md'):
        if (ROOT / name).is_file():
            entries[name] = ROOT / name
    pending = destination.with_suffix('.pending')
    hashes = {}
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, file in sorted(entries.items()):
            archive.write(file, name)
            hashes[name] = {'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
        archive.writestr('SOURCE_COMMIT.txt', state['source_commit'] + '\n')
        archive.writestr('MANIFEST.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(pending) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC check failed')
    pending.replace(destination)
    report = {'status': 'exported', 'zip': str(destination), 'bytes': destination.stat().st_size,
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'files': len(entries),
        'source_commit': state['source_commit'], 'campaign_status': state['status'], 'created_unix': time.time()}
    (OUT / 'archive.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (OUT / 'archive.json').write_text(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, indent=2))
        raise
