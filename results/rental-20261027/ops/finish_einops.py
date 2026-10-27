import subprocess
from pathlib import Path
root=Path('/root/gpufree-data/lockstep');python=str(root/'envs/nano_rl/bin/python')
subprocess.run([python,'-m','pip','install','--index-url','https://pypi.org/simple','einops'],check=True,timeout=180)
subprocess.run([python,'-m','pip','check'],check=True,timeout=30)
with (root/'envs/nano_rl/environment-freeze.txt').open('w') as log:subprocess.run([python,'-m','pip','freeze'],stdout=log,check=True,timeout=30)
print('Dependency closure passed; pinned environment ready for manual GPU smoke',flush=True)
