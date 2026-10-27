import concurrent.futures
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

root = Path('/root/gpufree-data/lockstep')
python = str(root / 'envs/nano_rl/bin/python')
pool = root / 'wheelhouse'
tmp = root / 'tmp'
tmp.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(tmp)
original = json.loads((root / 'results/bootstrap-01/gpu-env.status.json').read_text())
assert original['returncode'] == 124 and not original['cleanup']['surviving_pids']
metadata_url = 'https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl.metadata'
metadata = (root / 'results/bootstrap-01/torch-cached.metadata').read_text()
(root / 'results/bootstrap-01/torch-package.metadata').write_text(metadata)
requirements = []
for line in Parser().parsestr(metadata).get_all('Requires-Dist', []):
    req = Requirement(line)
    if not req.marker or req.marker.evaluate():
        requirements.append(Requirement(str(req).split(';')[0]))
requirements.append(Requirement('torch==2.8.0+cu128'))
existing = {}
for path in pool.glob('*.whl'):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
        meta = Parser().parsestr(archive.read(name).decode())
        existing[canonicalize_name(meta['Name'])] = meta['Version']
needed = [str(req) for req in requirements if canonicalize_name(req.name) not in existing
          or existing[canonicalize_name(req.name)] not in req.specifier]
needed.sort(key=lambda value: (0 if Requirement(value).name == 'torch' else 1 if Requirement(value).name.startswith('nvidia-') else 2 if Requirement(value).name == 'triton' else 3))
print(json.dumps({'needed': needed, 'reused': existing, 'metadata_source': 'complete original pip cache'}), flush=True)

def download(requirement):
    name = canonicalize_name(Requirement(requirement).name)
    path = root / 'results/bootstrap-01' / (name + '-download.log')
    with path.open('w') as log:
        result = subprocess.run([python, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
                                 '--dest', str(pool), '--index-url', 'https://download.pytorch.org/whl/cu128',
                                 '--progress-bar', 'off', requirement], stdout=log, stderr=subprocess.STDOUT, timeout=650)
    print(json.dumps({'download': requirement, 'returncode': result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError(f'download failed: {requirement}; inspect {path}')

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(download, needed))
wheels = list(pool.glob('*.whl'))
hashes = {}
for path in wheels:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    hashes[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
(root / 'results/bootstrap-01/dependency-wheel-hashes.json').write_text(json.dumps(hashes, indent=2))

def run(*args):
    subprocess.run([python, *args], cwd=root / 'repo', check=True, timeout=300)

run('-m', 'pip', 'install', '--no-deps', *map(str, wheels))
run('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', 'torch==2.8.0+cu128')
run('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', 'transformers==5.5.4',
    'packaging', 'psutil', 'wheel', 'setuptools>=77', 'numpy', 'xxhash', 'safetensors', 'tqdm',
    'matplotlib', 'datasets', 'ruff')
wheel = next((root / 'wheels').glob('flash_attn-*.whl'))
run('scripts/verify_gpu_wheel.py', '--wheel', str(wheel))
run('-m', 'pip', 'install', '--only-binary=:all:', '--no-deps', str(wheel))
run('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.')
run('-m', 'pip', 'check')
with (root / 'envs/nano_rl/environment-freeze.txt').open('w') as log:
    subprocess.run([python, '-m', 'pip', 'freeze'], stdout=log, check=True, timeout=30)
print('Pinned environment restored and verified; GPU kernels not yet run.', flush=True)
