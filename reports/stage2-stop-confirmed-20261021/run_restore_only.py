from pathlib import Path
import os,sys,hashlib,subprocess,json
root=Path('/data/matengyu/geyugong/mindhier-reproduction');out=root/'runs/s01-stage2-8500-restore-gate';source=root/'runs/recovery-source-06e95b8';original=source/'scripts/train_stage2_formal.py'
stop=json.loads((root/'runs/user-stop-8500-20261020/state.json').read_text());assert stop['phase']=='stopped_checkpoint_preserved'
assert not Path('/proc/20304').exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='06e95b83a14b07fef51ebdb4c299b6df9914bb34'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
os.chdir(source);sys.path.insert(0,str(source/'scripts'))
sys.argv=[str(original),'--run','s01-stage2-8500-restore-gate','--steps','9000','--microbatch','4','--resume']
exec(compile((out/'restore-only-source.py').read_text(),str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
