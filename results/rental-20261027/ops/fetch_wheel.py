import hashlib,json,urllib.request,time
from pathlib import Path
root=Path('/root/gpufree-data/lockstep')
settings=json.loads((root/'repo/configs/rental.json').read_text())
target=root/'wheels'/settings['wheel_filename']
part=target.with_suffix('.whl.part')
print('Fetching pinned official wheel',flush=True)
digest=hashlib.sha256(); total=0; marker=0
with urllib.request.urlopen(settings['wheel_url'],timeout=30) as incoming,part.open('wb') as outgoing:
 while True:
  chunk=incoming.read(1024*1024)
  if not chunk:break
  outgoing.write(chunk);digest.update(chunk);total+=len(chunk)
  if total-marker>=50*1024*1024:print('downloaded_bytes',total,flush=True);marker=total
if digest.hexdigest()!=settings['wheel_sha256']:raise ValueError('official wheel SHA256 mismatch')
part.replace(target)
print(json.dumps({'bytes':total,'sha256':digest.hexdigest(),'wheel':str(target)}),flush=True)
