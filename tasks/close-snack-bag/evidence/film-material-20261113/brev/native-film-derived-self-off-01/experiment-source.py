"""Native Newton VBD pouch with yield-based crease memory; uncalibrated diagnostic.
API pattern: Newton 1.7 example_cloth_hanging.py; plastic kernel: pinned RSS paper.py.
"""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import newton
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pouch_geometry import open_pouch
from film_material import FilmMaterial
from robosimstudio_materials.paper import _crease_kernel
from robosimstudio_materials.bag import layer_gap,mouth_opening
p=argparse.ArgumentParser()
p.add_argument("output",type=Path);p.add_argument("--device",default="cuda:0")
p.add_argument("--nx",type=int,default=40);p.add_argument("--ny",type=int,default=32)
p.add_argument("--material-profile",choices=("derived","legacy"),default="derived")
p.add_argument("--folded",action="store_true");p.add_argument("--yield-angle",type=float)
p.add_argument("--seconds",type=float,default=3);p.add_argument("--self-contact",type=int,choices=(0,1),default=1)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
wp.init();wp.set_device(a.device)
q,f,g=open_pouch(nx=a.nx,ny=a.ny,mouth_half_gap=.0003,foldable_mouth=True)
coeff=FilmMaterial().coefficients(min(.20/a.nx,.16/a.ny))
if a.material_profile=="legacy":coeff.update(areal_density=.08,tri_ke=1e4,tri_ka=1e4,edge_ke=3.,yield_angle=.45)
if a.yield_angle is not None:coeff["yield_angle"]=a.yield_angle
b=newton.ModelBuilder();b.add_ground_plane()
b.add_cloth_mesh(scale=1.0,pos=wp.vec3(0,0,.025),rot=wp.quat_identity(),vel=wp.vec3(0),
    vertices=[wp.vec3(*map(float,x)) for x in q],indices=f.ravel().tolist(),density=coeff["areal_density"],
    tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],tri_kd=.01,edge_ke=coeff["edge_ke"],edge_kd=.0001,particle_radius=.0001)
b.color(include_bending=True)
m=b.finalize(device=a.device);m.soft_contact_ke=1e4;m.soft_contact_kd=.01;m.soft_contact_mu=.3
solver=newton.solvers.SolverVBD(m,iterations=20,particle_enable_self_contact=bool(a.self_contact),
    particle_self_contact_margin=.001,particle_self_contact_gap=.0005,
    particle_topological_contact_filter_threshold=2,particle_rest_shape_contact_exclusion_radius=0.,
    particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128)
s0,s1=m.state(),m.state();ctrl=m.control()
if a.folded:
    from bag_surface_mesh import folded_positions
    initial=folded_positions(q+np.array([0.,0.,.019]),radius=.0018)-[0.,0.,.019]+[0.,0.,.025]
    s0.particle_q.assign(initial.astype(np.float32));s1.particle_q.assign(initial.astype(np.float32))
pipeline=newton.CollisionPipeline(m);contacts=pipeline.contacts()
edges=m.edge_indices.numpy();valid=np.all(edges>=0,axis=1)
mid=q[np.maximum(edges,0)].mean(axis=1)
rows=np.flatnonzero(valid & (mid[:,0]>=.14)).astype(np.int32)
rows_wp=wp.array(rows,dtype=wp.int32,device=a.device)
params=wp.array(np.array([coeff["yield_angle"],1.,coeff["recovery"]],dtype=np.float32),device=a.device)
entity=SimpleNamespace(top=g[1],bottom=g[0],spec=SimpleNamespace(cells=(a.nx,a.ny),
    material=SimpleNamespace(particle_radius=.0001)))
trace=[];snapshots={"q_00000":s0.particle_q.numpy().copy()}
started=time.monotonic();dt=1/600
for frame in range(round(a.seconds*60)):
    for _ in range(10):
        s0.clear_forces();pipeline.collide(s0,contacts)
        solver.step(s0,s1,ctrl,contacts,dt)
        wp.launch(_crease_kernel,dim=len(rows),inputs=[rows_wp,m.edge_indices,params,dt,s1.particle_q],
                  outputs=[m.edge_rest_angle],device=a.device)
        s0,s1=s1,s0
    measured=s0.particle_q.numpy().astype(float)
    gap=layer_gap(measured,entity)
    trace.append({"step":frame+1,"finite":bool(np.isfinite(measured).all()),
        "gap_m":float(gap) if np.isfinite(gap) else None,
        "max_speed_m_s":float(np.linalg.norm(s0.particle_qd.numpy(),axis=1).max())})
    if frame%15==0 or frame==round(a.seconds*60)-1:snapshots[f"q_{frame+1:05d}"]=measured.copy()
    if frame%30==0:print("PROGRESS",frame+1,trace[-1],flush=True)
result={"device":a.device,"self_contact":bool(a.self_contact),"initial_fold_imposed":a.folded,"material_profile":a.material_profile,"coefficients":coeff,"steps":len(trace),
  "wall_seconds":time.monotonic()-started,"physical_thickness_m":80e-6,
  "contact_radius_m":.0001,"mesh_cells":[a.nx,a.ny],"crease_hinges":len(rows),
  "all_finite":all(x["finite"] for x in trace),"mouth":mouth_opening(measured,entity),
  "limitation":"unforced native VBD pouch; homogeneous equivalent and uniform hinge width are assumptions; no air, food, clip or robot"}
(a.output/"result.json").write_text(json.dumps(result,indent=2))
(a.output/"trace.json").write_text(json.dumps(trace))
np.savez_compressed(a.output/"pouch-snapshots.npz",faces=f,**snapshots)
print("RESULT",json.dumps(result),flush=True)
