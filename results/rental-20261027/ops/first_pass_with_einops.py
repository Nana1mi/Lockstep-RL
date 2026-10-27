import json,os,subprocess,time
from pathlib import Path
root=Path('/root/gpufree-data/lockstep')
checks=['model','gpu-env-einops']
while time.time()<1793096040:
 states={}
 for name in checks:
  p=root/'results/bootstrap-01'/(name+'.status.json')
  states[name]=json.loads(p.read_text()) if p.exists() else None
 if any(value and value['returncode'] for value in states.values()):
  (root/'results/bootstrap-01/first-pass-wait-status.json').write_text(json.dumps({'status':'blocked_by_preparation','states':states},indent=2));raise SystemExit(1)
 if all(value and value['returncode']==0 for value in states.values()):break
 time.sleep(3)
else:
 (root/'results/bootstrap-01/first-pass-wait-status.json').write_text(json.dumps({'status':'insufficient_time_for_gpu_run'}));raise SystemExit(1)
python=str(root/'envs/nano_rl/bin/python')
remaining=max(1,int(1793096340-time.time()))
command=['/usr/bin/python3',str(root/'ops/run_command.py'),'rental-first-pass',str(remaining),python,'scripts/run_rental.py','--persistent-root',str(root),'--run-id','preflight-01','--python',python,'--stages','environment','baseline','--power-mode','L40S observed power limit350W; platform default clocks, no clock changes','--gpu-id','0','--allow-gpu']
(root/'results/bootstrap-01/first-pass-command.json').write_text(json.dumps({'command':command,'source':'3f59e33','work_deadline_utc':'2026-10-27T08:19:00Z'},indent=2))
raise SystemExit(subprocess.call(command,env=os.environ))
