import os,subprocess,sys
from pathlib import Path
root=Path('/root/gpufree-data/lockstep')
python=str(root/'py311/bin/python')
subprocess.run([python,'scripts/setup_managed_python.py','--persistent-root',str(root),'--provider','conda','--python',python,'--allow-install'],cwd=root/'repo',check=True,timeout=60)
wheel=next((root/'wheels').glob('flash_attn-*.whl'))
command=['bash','scripts/setup_gpu_env.sh','--allow-install','--env',str(root/'envs/nano_rl'),'--wheel',str(wheel),'--python',python,'--pip-index-url','https://pypi.org/simple']
raise SystemExit(subprocess.call(command,cwd=root/'repo'))
