import json,os,subprocess,time,sys
from pathlib import Path
root=Path('/root/gpufree-data/lockstep');boot=root/'results/bootstrap-01'
while time.time()<1793088845:time.sleep(2)
while time.time()<1793089050:
 if (boot/'first-pass-command.json').exists():raise SystemExit(0)
 p=boot/'gpu-env-resume3.status.json'
 if p.exists():
  if json.loads(p.read_text())['returncode']:raise SystemExit(1)
  break
 time.sleep(2)
else:raise SystemExit(1)
python=str(root/'envs/nano_rl/bin/python')
cmd=['/usr/bin/python3',str(root/'ops/run_command.py'),'late-environment',str(max(1,int(1793089140-time.time()))),python,'scripts/run_rental.py','--persistent-root',str(root),'--run-id','preflight-late-environment-01','--python',python,'--stages','environment','--power-mode','L40S observed power limit350W; platform default clocks','--gpu-id','0','--allow-gpu']
(boot/'late-environment-command.json').write_text(json.dumps({'command':cmd,'scope':'Only environment/attention; full baseline not claimed'},indent=2))
code=subprocess.call(cmd,env=os.environ)
if code or 1793089140-time.time()<90:raise SystemExit(code)
sys.path.insert(0,str(root/'repo'))
from scripts.run_rental import build_plan
from lockstep_rl.rendezvous import check_port
plan=build_plan(root,'preflight-late-environment-01',python,'L40S power limit350W; default clocks',['environment'])
config=root/'results/preflight-late-environment-01/single-eager-config.json'
config.write_text(json.dumps(plan['configs']['baseline-smoke'],indent=2))
env={**os.environ,'LOCKSTEP_NCCL_PORT':str(check_port(0))}
cmd=['/usr/bin/python3',str(root/'ops/run_command.py'),'single-current-eager',str(max(1,int(1793089140-time.time()))),python,'scripts/inference_benchmark_worker.py','--allow-gpu','--config',str(config),'--case','short-single','--source','current','--source-path',str(root/'repo'),'--backend','eager','--output',str(config.parent/'single-current-eager.json')]
(boot/'single-eager-command.json').write_text(json.dumps({'command':cmd,'scope':'Single-process smoke only; not a repeated upstream/current contrast'},indent=2))
raise SystemExit(subprocess.call(cmd,env=env))
