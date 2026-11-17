"""Load through YAM's public entry point and exercise raw-sample score changes."""
from dataclasses import asdict
from pathlib import Path
import sys
from yam_arena.task_file import load

task = load(Path(sys.argv[1]))
module = task.module
Sample = module._grading["Sample"]
base = asdict(Sample(90,0,0,0,0,2,0,0,0,0,0,0,4,30,.04,True,True,True,True,True,False))
for hz in (240,480):
    module.reset_native_observations(hz)
    for tick in range(1,hz+1):
        obs = dict(base,physics_tick=tick,opening_force=0 if tick<=hz//2 else 2)
        module.publish_native_observation(obs)
    assert task.score({})==.5
    assert module.publish_native_observation(dict(obs,physics_tick=tick+1,screw_height=.9e-6))==.5
    assert module.publish_native_observation(dict(obs,physics_tick=tick+2,screw_height=1.1e-6))==0
    assert module.publish_native_observation(dict(obs,physics_tick=tick+3))==0
    assert not task.success({})
    module.reset_native_observations(hz)
    try:
        module.publish_native_observation({"score":1., "hard_failure":False})
    except KeyError:
        pass
    else:
        raise AssertionError("unmeasured submission was not rejected")
    assert task.score({})==0
    print(f"{hz} Hz common task loader: baseline, partial, normal retention, fault, reset, missing telemetry passed")
