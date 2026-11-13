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
p.add_argument("--prepare-only",action="store_true")
p.add_argument("--clip",action="store_true",help="Free spring clip initially open around the fold; closing-only diagnostic")
p.add_argument("--folded",action="store_true");p.add_argument("--yield-angle",type=float)
p.add_argument("--seconds",type=float,default=3);p.add_argument("--self-contact",type=int,choices=(0,1),default=1)
a=p.parse_args()
if a.seconds<=0 and not a.prepare_only:p.error("positive duration required")
a.output.mkdir(parents=True,exist_ok=False)
wp.init();wp.set_device(a.device)
q,f,g=open_pouch(nx=a.nx,ny=a.ny,mouth_half_gap=.0003,foldable_mouth=True)
coeff=FilmMaterial().coefficients(min(.20/a.nx,.16/a.ny))
if a.material_profile=="legacy":coeff.update(areal_density=.08,tri_ke=1e4,tri_ka=1e4,edge_ke=3.,yield_angle=.45)
if a.yield_angle is not None:coeff["yield_angle"]=a.yield_angle
b=newton.ModelBuilder()
if a.clip:b.default_shape_cfg.gap=.0005
b.add_ground_plane()
b.add_cloth_mesh(scale=1.0,pos=wp.vec3(0,0,.025),rot=wp.quat_identity(),vel=wp.vec3(0),
    vertices=[wp.vec3(*map(float,x)) for x in q],indices=f.ravel().tolist(),density=coeff["areal_density"],
    tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],tri_kd=.01,edge_ke=coeff["edge_ke"],edge_kd=.0001,particle_radius=.0001)
clip_root = clip_upper = None
if a.clip:
    asset=Path(__file__).resolve().parents[1]/"assets/clip_v5/clip_v5.usda"
    b.add_usd(str(asset),floating=True,xform=wp.transform(wp.vec3(.184,0,.0252),wp.quat(0,0,1,0)),
              override_root_xform=True)
    hinge=next(i for i,label in enumerate(b.joint_label) if label.endswith("clip_hinge"))
    dof=b.joint_qd_start[hinge]
    b.joint_q[b.joint_q_start[hinge]]=-.55
    b.joint_target_q[b.joint_q_start[hinge]]=0.
    b.joint_target_ke[dof]=.13
    b.joint_target_kd[dof]=.003
    b.joint_target_mode[dof]=newton.JointTargetMode.POSITION
    clip_root,clip_upper=b.joint_parent[hinge],b.joint_child[hinge]
b.color(include_bending=True)
m=b.finalize(device=a.device);m.soft_contact_ke=1e4;m.soft_contact_kd=.01;m.soft_contact_mu=.3
solver=newton.solvers.SolverVBD(m,iterations=50 if a.clip else 20,rigid_body_particle_contact_buffer_size=16384 if a.clip else 256,rigid_compliant_alm=True,particle_enable_self_contact=bool(a.self_contact),
    particle_self_contact_margin=.001,particle_self_contact_gap=.0005,
    particle_topological_contact_filter_threshold=2,particle_rest_shape_contact_exclusion_radius=0.,
    particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128)
s0,s1=m.state(),m.state();ctrl=m.control()
if a.clip:
    newton.eval_fk(m,m.joint_q,m.joint_qd,s0)
    newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
if a.folded:
    from bag_surface_mesh import folded_positions
    initial=folded_positions(q+np.array([0.,0.,.019]),radius=.0018)-[0.,0.,.019]+[0.,0.,.025]
    s0.particle_q.assign(initial.astype(np.float32));s1.particle_q.assign(initial.astype(np.float32))
pipeline=newton.CollisionPipeline(m,enable_rigid_soft_full_surface_contact=a.clip,
    **({"soft_contact_gap":.0005} if a.clip else {}));contacts=pipeline.contacts()
observables=solver.observables({"contact_f"},contacts=contacts) if a.clip else None
edges=m.edge_indices.numpy();valid=np.all(edges>=0,axis=1)
mid=q[np.maximum(edges,0)].mean(axis=1)
rows=np.flatnonzero(valid & (mid[:,0]>=.14)).astype(np.int32)
rows_wp=wp.array(rows,dtype=wp.int32,device=a.device)
params=wp.array(np.array([coeff["yield_angle"],1.,coeff["recovery"]],dtype=np.float32),device=a.device)
entity=SimpleNamespace(top=g[1],bottom=g[0],spec=SimpleNamespace(cells=(a.nx,a.ny),
    material=SimpleNamespace(particle_radius=.0001)))
rest_initial=m.edge_rest_angle.numpy().copy()
trace=[];snapshots={"q_00000":s0.particle_q.numpy().copy()}
if a.clip:
    snapshots["b_00000"]=s0.body_q.numpy().copy()
    (a.output/"clip-shapes.json").write_text(json.dumps({"body_labels":list(m.body_label),
        "shape_body":m.shape_body.numpy().tolist(),"shape_local":m.shape_transform.numpy().tolist(),
        "shape_scale":m.shape_scale.numpy().tolist(),"shape_type":m.shape_type.numpy().tolist(),
        "root":clip_root,"upper":clip_upper},indent=2))
if a.clip:
    from check_native_clip import inspect_state
    shape=json.loads((a.output/"clip-shapes.json").read_text())
    preflight=inspect_state(s0.particle_q.numpy(),f,s0.body_q.numpy(),shape)
    (a.output/"preflight.json").write_text(json.dumps(preflight,indent=2))
    np.savez_compressed(a.output/"initial-snapshots.npz",faces=f,**snapshots)
    print("PREFLIGHT",preflight,flush=True)
    if any(x["penetrating_triangles"] for x in preflight):
        raise RuntimeError("initial clip/film intrusion: reject before stepping")
if a.prepare_only:
    np.savez_compressed(a.output/"pouch-snapshots.npz",faces=f,**snapshots)
    sys.exit(0)
started=time.monotonic();dt=1/600
for frame in range(round(a.seconds*60)):
    for _ in range(10):
        s0.clear_forces();pipeline.collide(s0,contacts)
        solver.step(s0,s1,ctrl,contacts,dt,observables=observables)
        wp.launch(_crease_kernel,dim=len(rows),inputs=[rows_wp,m.edge_indices,params,dt,s1.particle_q],
                  outputs=[m.edge_rest_angle],device=a.device)
        s0,s1=s1,s0
    measured=s0.particle_q.numpy().astype(float)
    gap=layer_gap(measured,entity)
    trace.append({"step":frame+1,"finite":bool(np.isfinite(measured).all()),
        "gap_m":float(gap) if np.isfinite(gap) else None,
        "max_speed_m_s":float(np.linalg.norm(s0.particle_qd.numpy(),axis=1).max())})
    if a.clip:
        from scipy.spatial.transform import Rotation
        body=s0.body_q.numpy()
        jaw=(Rotation.from_quat(body[clip_root,3:]).inv()*Rotation.from_quat(body[clip_upper,3:])).as_rotvec()[1]
        n=min(int(contacts.soft_contact_count.numpy()[0]),contacts.soft_contact_max)
        ids=contacts.soft_contact_shape.numpy()[:n]
        forces=observables.contact_f.numpy()[contacts.rigid_contact_max:contacts.rigid_contact_max+n,:3]
        shape_body=m.shape_body.numpy()
        sums={str(i):float(np.linalg.norm(forces[shape_body[ids]==i],axis=1).sum()) for i in (clip_root,clip_upper)}
        nets={str(i):float(np.linalg.norm(forces[shape_body[ids]==i].sum(axis=0))) for i in (clip_root,clip_upper)}
        trace[-1].update(clip_root_pose=body[clip_root].tolist(),jaw_angle_rad=float(jaw),
            clip_soft_force_norm_sum_N=sums,clip_soft_net_force_N=nets,soft_contact_count=n)
    if frame%15==0 or frame==round(a.seconds*60)-1:
        snapshots[f"q_{frame+1:05d}"]=measured.copy()
        if a.clip:snapshots[f"b_{frame+1:05d}"]=body.copy()
    if frame%30==0:print("PROGRESS",frame+1,trace[-1],flush=True)
rest_final=m.edge_rest_angle.numpy()
crease_delta=np.abs(np.arctan2(np.sin(rest_final-rest_initial),np.cos(rest_final-rest_initial)))[rows]
result={"residual_rest_angle_max_rad":float(crease_delta.max()),"residual_rest_angle_over_0_05_count":int((crease_delta>.05).sum()),"device":a.device,"self_contact":bool(a.self_contact),"initial_fold_imposed":a.folded,"material_profile":a.material_profile,"coefficients":coeff,"steps":len(trace),
  "wall_seconds":time.monotonic()-started,"physical_thickness_m":80e-6,
  "contact_radius_m":.0001,"mesh_cells":[a.nx,a.ny],"crease_hinges":len(rows),
  "all_finite":all(x["finite"] for x in trace),"mouth":mouth_opening(measured,entity),
  "clip":a.clip,"limitation":"homogeneous equivalent and uniform hinge width are assumptions; optional clip starts open around fold; no insertion, air, food or robot"}
(a.output/"result.json").write_text(json.dumps(result,indent=2))
(a.output/"trace.json").write_text(json.dumps(trace))
np.savez_compressed(a.output/"pouch-snapshots.npz",faces=f,edges=edges,rest_initial=rest_initial,rest_final=rest_final,**snapshots)
print("RESULT",json.dumps(result),flush=True)
