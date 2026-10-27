import json,os,subprocess,time
from pathlib import Path
root=Path('/root/gpufree-data/lockstep')
status=root/'results/bootstrap-01/hf-bootstrap.status.json'
end=time.monotonic()+270
while not status.exists():
 if time.monotonic()>end:raise TimeoutError('HF bootstrap not ready')
 time.sleep(2)
if json.loads(status.read_text())['returncode']:raise RuntimeError('HF bootstrap failed')
env={**os.environ,'PYTHONPATH':str(root/'bootstrap-deps'),'HTTP_PROXY':'http://127.0.0.1:17897','HTTPS_PROXY':'http://127.0.0.1:17897','HF_HUB_DISABLE_XET':'1','HF_HOME':str(root/'cache/huggingface')}
command=['/usr/bin/python3',str(root/'ops/run_command.py'),'model', '900','/opt/conda/bin/python',str(root/'repo/scripts/download_model.py'),'--allow-download','--hf-endpoint','https://huggingface.co','--destination',str(root/'models/Qwen3-0.6B')]
raise SystemExit(subprocess.call(command,env=env))
