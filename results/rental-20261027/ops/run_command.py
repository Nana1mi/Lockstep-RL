import json,os,sys
from pathlib import Path
sys.path.insert(0,'/root/gpufree-data/lockstep/repo')
from lockstep_rl.process_runner import run_task
root=Path('/root/gpufree-data/lockstep')
name=sys.argv[1]
env={**os.environ,'CONDA_PKGS_DIRS':str(root/'cache/conda-pkgs'),'CONDA_ENVS_PATH':str(root/'envs'),'PIP_CACHE_DIR':str(root/'cache/pip'),'HF_HOME':str(root/'cache/huggingface'),'XDG_CACHE_HOME':str(root/'cache'),'UV_CACHE_DIR':str(root/'cache/uv'),'TORCHINDUCTOR_CACHE_DIR':str(root/'cache/torchinductor'),'TRITON_CACHE_DIR':str(root/'cache/triton'),'HF_HUB_DISABLE_XET':'1','LOCKSTEP_TASK_DEADLINE':'1793096340'}
if os.environ.get('LOCKSTEP_USE_PROXY')=='1':
 env.update(HTTP_PROXY='http://127.0.0.1:17897',HTTPS_PROXY='http://127.0.0.1:17897',http_proxy='http://127.0.0.1:17897',https_proxy='http://127.0.0.1:17897',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
with (root/'results/bootstrap-01'/f'{name}.log').open('w') as out:
 result=run_task(sys.argv[3:],timeout_seconds=float(sys.argv[2]),stdout=out,env=env,cwd=root/'repo')
(root/'results/bootstrap-01'/f'{name}.status.json').write_text(json.dumps(result.record(),indent=2))
raise SystemExit(result.returncode)
