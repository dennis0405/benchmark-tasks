"""CPU VBD film-strip crease diagnostic; fixtures pin nodes, no robot claim."""
import json,sys,time
from pathlib import Path
from unittest import mock
import numpy as np
import robosimstudio as rss
import robosimstudio_materials.paper as paper
from tests import test_paper_crease as upstream
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
film=rss.replace(paper.PAPER,thickness=80e-6,areal_density=.08,particle_radius=.0005,
                 tri_ke=1e4,tri_ka=1e4,edge_ke=3.0,plastic_rate=1.0,plastic_recovery=.1)
physics=rss.GENERIC.with_backend("paper",rss.backend("newton.solvers.SolverVBD", iterations=20, particle_enable_self_contact=True, particle_self_contact_radius=.001, particle_self_contact_margin=.003, particle_topological_contact_filter_threshold=2, particle_rest_shape_contact_exclusion_radius=0.0, particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128))
upstream.LENGTH=.20;upstream.WIDTH=.05;upstream.CELL=.01
upstream.CARRY=90;upstream.HOLD=30;upstream.PRESS=30;upstream.FREE=180;upstream.PRESS_TO=.004
real_build=rss.Scene.build
for yield_angle in (.45,0.0):
    name="plastic" if yield_angle else "elastic"
    snapshots={};counter=[0];faces=[None];started=time.monotonic()
    def build(scene,*args,**kwargs):
        sim=real_build(scene,*args,**{**kwargs,"device":"cpu"})
        faces[0]=sim.model.tri_indices.numpy()
        solver=sim.pipeline.program.solver_for("strip")
        print("SOLVER",type(solver).__name__,"self_contact",getattr(solver,"particle_enable_self_contact",None),flush=True)
        real_run=sim.run
        def run(count=1):
            for _ in range(count):
                real_run(1)
                counter[0]+=1
                if counter[0]%15==0:
                    snapshots[f"q_{counter[0]:05d}"]=sim.state.particle_q.numpy().copy()
                if counter[0]%60==0:print("PROGRESS",name,counter[0],flush=True)
            return sim
        sim.run=run
        return sim
    print("BEGIN",name,flush=True)
    with mock.patch.object(rss,"GENERIC",physics),mock.patch.object(paper,"PAPER",film),mock.patch.object(rss.Scene,"build",build):
        result=upstream._fold_a_strip(yield_angle)
    result.update(yield_angle=yield_angle,plastic_recovery_per_s=.1,physical_thickness_m=80e-6,
        wall_seconds=time.monotonic()-started,device="cpu",
        limitation="numerical film-strip hypothesis with pinned fixtures; not bag, physical gripper or calibrated film")
    (out/(name+".json")).write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/(name+"-snapshots.npz"),faces=faces[0],**snapshots)
    print("RESULT",name,json.dumps(result),flush=True)
