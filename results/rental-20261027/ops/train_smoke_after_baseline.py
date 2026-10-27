import json,os,subprocess,time,sys
from pathlib import Path
root=Path('/root/gpufree-data/lockstep');boot=root/'results/bootstrap-01';status=boot/'rental-first-pass.status.json'
while not status.exists():
 if time.time()>1793095140:raise TimeoutError('baseline did not finish within readiness budget')
 time.sleep(3)
state=json.loads(status.read_text())
if state.get('cleanup',{}).get('surviving_pids'):raise RuntimeError('baseline cleanup incomplete')
manifest=json.loads((root/'results/preflight-01/manifest.json').read_text())
gates={j['name']:j['returncode'] for j in manifest['completed_jobs']}
if gates.get('environment')!=0 or gates.get('attention')!=0:raise RuntimeError('hard GPU gates failed')
settings=json.loads((root/'repo/configs/train-smoke-simple.json').read_text())
settings.update(model=str(root/'models/Qwen3-0.6B'),iterations=2,prompts_per_iteration=4,learning_rate=5e-6,reward_parser='lenient',train_count=512,validation_count=8,sync_probe_tokens=16,validation_status='MANUAL_GPU_SMOKE; runtime configuration, not a learning claim')
settings['grpo'].update(group_size=8,microbatch_size=4,gradient_accumulation_steps=8,update_steps=2,anchor='recomputed')
folder=root/'configs';folder.mkdir(exist_ok=True);config=folder/'readiness-grpo.json';config.write_text(json.dumps(settings,indent=2))
sys.path.insert(0,str(root/'repo'))
from lockstep_rl.rendezvous import check_port
env={**os.environ,'LOCKSTEP_NCCL_PORT':str(check_port(0))}
python=str(root/'envs/nano_rl/bin/python')
cmd=['/usr/bin/python3',str(root/'ops/run_command.py'),'readiness-grpo','1200',python,'scripts/train_grpo.py','--allow-gpu','--config',str(config),'--anchor','recomputed','--output-dir',str(root/'results/readiness-grpo-01')]
(boot/'readiness-grpo-command.json').write_text(json.dumps({'command':cmd,'baseline_returncode':state['returncode'],'scope':'2 iterations,4 prompts,G8,2 updates; execution/sync readiness only'},indent=2))
raise SystemExit(subprocess.call(cmd,env=env))
