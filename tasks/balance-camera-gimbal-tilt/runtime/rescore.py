"""Replay measured states with a task-defined scorer, without starting Isaac Sim."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from yam_arena.task_file import ObjectState

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',type=Path,required=True)
p.add_argument('trajectory',type=Path)
p.add_argument('--expect',type=Path)
a=p.parse_args()
sys.path.insert(0,str(a.task.resolve().parent))
spec=importlib.util.spec_from_file_location('replayed_task',a.task.resolve())
task=importlib.util.module_from_spec(spec);sys.modules[spec.name]=task;spec.loader.exec_module(task)
scorer=task.make_scorer();result=None
for line in a.trajectory.open():
    row=json.loads(line)
    states={n:ObjectState(**{k:np.asarray(v,dtype=float) for k,v in obj.items()}) for n,obj in row['objects'].items()}
    result=scorer.update(step=row['step'],objects=states,root_poses={n:np.asarray(v) for n,v in row['root_poses'].items()},robot=row['robot'],
                        **({'mechanics':row['mechanics']} if 'mechanics' in row else {}))
if result is None:raise SystemExit('Empty trajectory')
summary={'steps':row['step'],'score':result['score'],'success':result.get('success',result['score']>=1)}
if a.expect:
    expected=json.loads(a.expect.read_text())
    for k in summary:
        if summary[k]!=expected[k]:raise SystemExit('Replay mismatch: '+k)
print(json.dumps(summary,indent=2))
