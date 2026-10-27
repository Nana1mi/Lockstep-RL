"""Restricted read-only follow-up snapshot and completed-archive download."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/gpufree-data/lockstep')
OUT = ROOT / 'results/followup-ce9e354-01'
def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return {'returncode': p.returncode, 'stdout': p.stdout[-12000:], 'stderr': p.stderr[-1000:]}
    except Exception as exc:
        return {'error': str(exc)}
def snapshot():
    state=read(OUT/'manifest.json')
    disk=shutil.disk_usage(ROOT)
    cg=Path('/sys/fs/cgroup')
    answer={'unix_time':time.time(),'campaign':state,'archive':read(OUT/'archive.json'),
        'source_head':command(['git','-C',str(ROOT/'repo-ce9e354'),'rev-parse','HEAD']),
        'source_tracked_status':command(['git','-C',str(ROOT/'repo-ce9e354'),'status','--porcelain','--untracked-files=no']),
        'gpu':command(['nvidia-smi','--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv']),
        'container_memory':{'limit_bytes':int((cg/'memory.max').read_text()),'usage_including_cache_bytes':int((cg/'memory.current').read_text())},
        'container_cpu_max':(cg/'cpu.max').read_text().strip(),'disk':{'free_bytes':disk.free,'total_bytes':disk.total},'training':{}}
    for job in state.get('plan',{}).get('jobs',[]):
        folder=OUT/job['name']
        path=folder/'metrics.csv'
        if path.exists():
            with path.open() as f:
                rows=list(csv.DictReader(f))
            last=rows[-1] if rows else {}
            answer['training'][job['name']]={'iterations_recorded':len(rows),'mtime_unix':path.stat().st_mtime,
                'latest':last,'summary':read(folder/'summary.json')}
    return answer
def main():
    if os.environ.get('SSH_ORIGINAL_COMMAND') == 'download-results':
        info=read(OUT/'archive.json')
        target=ROOT/'exports/followup-ce9e354-01-results.zip'
        if info.get('status')!='exported' or not target.is_file():
            raise SystemExit('Export is not ready')
        with target.open('rb') as stream:
            shutil.copyfileobj(stream,sys.stdout.buffer,1024*1024)
        return
    p=argparse.ArgumentParser()
    p.add_argument('--watch',type=float,default=0)
    args=p.parse_args()
    while True:
        print(json.dumps(snapshot(),indent=2),flush=True)
        if not args.watch:
            return
        time.sleep(max(2,args.watch))
if __name__=='__main__':
    main()