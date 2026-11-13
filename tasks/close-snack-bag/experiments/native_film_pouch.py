"""Native Newton VBD pouch with yield-based crease memory; uncalibrated diagnostic.
API pattern: Newton 1.7 example_cloth_hanging.py; plastic kernel: pinned RSS paper.py.
"""
import argparse,json,sys,time,shutil
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
p.add_argument("--compliant-jaw",action="store_true",help="Five independently spring-loaded upper pads")
p.add_argument("--pad-stiffness",type=float,default=50.)
p.add_argument("--pad-target",type=float,default=-.002)
p.add_argument("--pad-damping",type=float,default=.5)
p.add_argument("--folded",action="store_true");p.add_argument("--yield-angle",type=float)
p.add_argument("--seconds",type=float,default=3);p.add_argument("--self-contact",type=int,choices=(0,1),default=1)
p.add_argument("--verify-every-step",action="store_true")
p.add_argument("--topology-filter",type=int,default=2,help="Self-contact ring exclusion; 1 retains only incident-primitive filtering")
p.add_argument("--record-every",type=int,default=15)
p.add_argument("--substeps",type=int,default=10)
p.add_argument("--iterations",type=int,help="VBD iterations per physics step")
p.add_argument("--collision-refresh",type=int,default=0,help="Owned contact detection every N iterations; -1 before/after initialization; zero historical external detection")
p.add_argument("--rigid-soft-dat",action="store_true",help="Truncate rigid/film motion against detected division planes; needs owned collision refresh")
p.add_argument("--clip-offset",type=float,nargs=3,default=(0.,0.,0.))
p.add_argument("--initial-open-angle",type=float,default=-.55)
p.add_argument("--open-limit",type=float,default=-.55)
p.add_argument("--mouth-drop",type=float,default=0.,help="Resting pouch neck descends over x=120..140 mm, metres")
p.add_argument("--spring-damping",type=float,default=.003)
p.add_argument("--spring-stiffness",type=float,default=.13)
p.add_argument("--contact-stiffness",type=float,default=1e4)
p.add_argument("--rigid-contact-stiffness",type=float,help="Override all rigid shape contact stiffness, including ground")
p.add_argument("--clip-contact-margin",type=float,default=0.,help="Conservative numerical clip contact skin, metres; geometry checks still use actual box surfaces")
p.add_argument("--ground-contact-margin",type=float,default=0.,help="Conservative numerical plane skin, metres; geometry checks still use z=0")
p.add_argument("--stop-on-geometry-error",action="store_true",help="Save the first failing physics state and reject early")
p.add_argument("--cuda-graph",action="store_true",help="Capture a control frame; retain checks at every physics step")
a=p.parse_args()
if a.seconds<=0 and not a.prepare_only:p.error("positive duration required")
if a.substeps<1 or a.record_every<1:p.error("substeps and record-every must be positive")
if a.rigid_soft_dat and a.collision_refresh==0:p.error("rigid-soft-dat requires owned collision refresh")
if a.compliant_jaw and not a.clip:p.error("compliant-jaw needs clip")
if a.cuda_graph and (not a.verify_every_step or a.substeps%2 or a.stop_on_geometry_error or not a.device.startswith('cuda')):
    p.error("cuda-graph requires CUDA, even substeps, verify-every-step, and no stop-on-geometry-error")
a.output.mkdir(parents=True,exist_ok=False)
source_dir=a.output/"sources";source_dir.mkdir()
for name in (Path(__file__),Path(__file__).with_name("gpu_contact_checks.py"),
             Path(__file__).with_name('check_native_clip.py'),
             Path(__file__).with_name("film_material.py"),Path(__file__).resolve().parents[1]/"pouch_geometry.py",
             Path(__file__).resolve().parents[1]/"bag_surface_mesh.py"):
    shutil.copy2(name,source_dir/name.name)
if a.compliant_jaw:
    shutil.copy2(Path(__file__).with_name('assets')/'clip_compliant_v1.xml',source_dir/'clip_compliant_v1.xml')
(a.output/"arguments.json").write_text(json.dumps(vars(a),default=str,indent=2))
wp.init();wp.set_device(a.device)
q,f,g=open_pouch(nx=a.nx,ny=a.ny,mouth_half_gap=.0003,foldable_mouth=True)
undeformed_neck=q.copy()
neck_t=np.clip((q[:,0]-.12)/.02,0.,1.)
neck_drop=a.mouth_drop*(neck_t*neck_t*(3.-2.*neck_t))
q[:,2]-=neck_drop
coeff=FilmMaterial().coefficients(min(.20/a.nx,.16/a.ny))
if a.material_profile=="legacy":coeff.update(areal_density=.08,tri_ke=1e4,tri_ka=1e4,edge_ke=3.,yield_angle=.45)
if a.yield_angle is not None:coeff["yield_angle"]=a.yield_angle
b=newton.ModelBuilder()
if a.clip:b.default_shape_cfg.gap=.0005
b.add_ground_plane()
b.shape_margin[0]=a.ground_contact_margin
b.add_cloth_mesh(scale=1.0,pos=wp.vec3(0,0,.025),rot=wp.quat_identity(),vel=wp.vec3(0),
    vertices=[wp.vec3(*map(float,x)) for x in q],indices=f.ravel().tolist(),density=coeff["areal_density"],
    tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],tri_kd=.01,edge_ke=coeff["edge_ke"],edge_kd=.0001,particle_radius=.0001)
clip_root = clip_upper = None
jaw_groups=[]
if a.clip:
    asset=Path(__file__).resolve().parents[1]/"assets/clip_v5/clip_v5.usda"
    clip_pos=np.array([.184,0,.0252])+a.clip_offset
    clip_xform=wp.transform(wp.vec3(*clip_pos),wp.quat(0,0,1,0))
    if a.compliant_jaw:
        b.add_mjcf(str(Path(__file__).with_name('assets')/'clip_compliant_v1.xml'),
            floating=True,xform=clip_xform,override_root_xform=True,parse_visuals=False)
    else:
        b.add_usd(str(asset),floating=True,xform=clip_xform,override_root_xform=True)
    for sid,owner in enumerate(b.shape_body):
        if owner>=0:b.shape_margin[sid]=a.clip_contact_margin
    hinge=next(i for i,label in enumerate(b.joint_label) if label.endswith("clip_hinge"))
    dof=b.joint_qd_start[hinge]
    b.joint_limit_lower[dof]=a.open_limit
    if a.compliant_jaw:b.joint_limit_upper[dof]=0.
    b.joint_target_q[b.joint_q_start[hinge]]=0.
    b.joint_target_ke[dof]=a.spring_stiffness
    b.joint_target_kd[dof]=a.spring_damping
    b.joint_target_mode[dof]=newton.JointTargetMode.POSITION
    clip_root,clip_upper=b.joint_parent[hinge],b.joint_child[hinge]
    lower_jaw=[sid for sid,owner in enumerate(b.shape_body) if owner==clip_root and wp.transform_get_translation(b.shape_transform[sid])[0]>0]
    if a.compliant_jaw:
        pad_joints=[j for j,label in enumerate(b.joint_label) if 'pad_slide_' in label]
        if len(pad_joints)!=5:raise RuntimeError('expected five compliant pad joints')
        pad_bodies=[b.joint_child[j] for j in pad_joints]
        for j in pad_joints:
            qi,di=b.joint_q_start[j],b.joint_qd_start[j]
            if abs(b.joint_q[qi])>1e-7:raise RuntimeError('pad authored rest pose must be zero')
            b.joint_target_q[qi]=a.pad_target;b.joint_target_ke[di]=a.pad_stiffness;b.joint_target_kd[di]=a.pad_damping
            b.joint_target_mode[di]=newton.JointTargetMode.POSITION
            b.joint_limit_ke[di]=1e6
        upper_jaw=[sid for sid,owner in enumerate(b.shape_body) if owner in pad_bodies]
    else:
        upper_jaw=[sid for sid,owner in enumerate(b.shape_body) if owner==clip_upper and wp.transform_get_translation(b.shape_transform[sid])[0]>0]
    jaw_groups=[lower_jaw,upper_jaw]
if a.rigid_contact_stiffness is not None:
    b.shape_material_ke[:]=[a.rigid_contact_stiffness]*len(b.shape_material_ke)
b.color(include_bending=True)
m=b.finalize(device=a.device);m.soft_contact_ke=a.contact_stiffness;m.soft_contact_kd=.01;m.soft_contact_mu=.3
pipeline_args={"soft_contact_gap":.0005} if a.clip else {}
if a.collision_refresh!=0:pipeline_args['contact_matching']='latest'
pipeline=newton.CollisionPipeline(m,enable_rigid_soft_full_surface_contact=a.clip,**pipeline_args)
owned={}
if a.collision_refresh!=0:
    Slot=newton.solvers.SolverBase.CollisionSlot
    Frequency=newton.solvers.SolverBase.CollisionFrequencyType
    detection_type=Frequency.PRE_POST_INIT if a.collision_refresh<0 else Frequency.ITERATIONS
    detection_frequency=max(1,a.collision_refresh)
    frequencies={Slot.RIGID:detection_frequency};frequency_types={Slot.RIGID:detection_type}
    if a.rigid_soft_dat:
        frequencies[Slot.SOFT_SELF_CONTACT]=detection_frequency
        frequency_types[Slot.SOFT_SELF_CONTACT]=detection_type
    owned=dict(collision_pipeline=pipeline,collision_frequency=frequencies,
        collision_frequency_type=frequency_types,rigid_soft_enable_dat=a.rigid_soft_dat)
iterations=a.iterations if a.iterations is not None else (50 if a.clip else 20)
solver=newton.solvers.SolverVBD(m,iterations=iterations,rigid_body_particle_contact_buffer_size=16384 if a.clip else 256,rigid_compliant_alm=True,particle_enable_self_contact=bool(a.self_contact),
    particle_self_contact_margin=.001,particle_self_contact_gap=.0005,
    particle_topological_contact_filter_threshold=a.topology_filter,particle_rest_shape_contact_exclusion_radius=0.,
    particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128,**owned)
s0,s1=m.state(),m.state();ctrl=m.control()
if a.clip:
    # Preserve authored rest poses for the solver. Open only the episode state.
    initial_joints=m.joint_q.numpy().copy()
    initial_joints[b.joint_q_start[hinge]]=a.initial_open_angle
    initial_joints_wp=wp.array(initial_joints,dtype=wp.float32,device=a.device)
    for state in (s0,s1):
        if state.joint_q is not None:state.joint_q.assign(initial_joints)
        newton.eval_fk(m,initial_joints_wp,m.joint_qd,state)
if a.folded:
    from bag_surface_mesh import folded_positions
    initial=folded_positions(undeformed_neck+np.array([0.,0.,.019]),radius=.0018)-[0.,0.,.019]+[0.,0.,.025]
    initial[:,2]-=neck_drop
    s0.particle_q.assign(initial.astype(np.float32));s1.particle_q.assign(initial.astype(np.float32))
contacts=solver.contacts if owned else pipeline.contacts()
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
        "shape_margin":m.shape_margin.numpy().tolist(),
        "shape_contact_stiffness":m.shape_material_ke.numpy().tolist(),
        "jaw_shape_groups":jaw_groups,
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
started=time.monotonic();dt=1/(60*a.substeps)
geometry=None
if a.verify_every_step:
    from gpu_contact_checks import GeometryChecks
    geometry=GeometryChecks(m,(s0,s1),f,(a.substeps if a.cuda_graph else round(a.seconds*60)*a.substeps)+1,ground_plane=True)
    geometry.check(s0,0)
physics_step=0
fold_ids=np.flatnonzero(q[:,0]>=.16)
def advance_substep(slot):
    global s0,s1
    s0.clear_forces()
    if not owned:pipeline.collide(s0,contacts)
    solver.step(s0,s1,ctrl,None if owned else contacts,dt,observables=observables)
    wp.launch(_crease_kernel,dim=len(rows),inputs=[rows_wp,m.edge_indices,params,dt,s1.particle_q],
              outputs=[m.edge_rest_angle],device=a.device)
    s0,s1=s1,s0
    if geometry:geometry.check(s0,slot)

buffers={} if geometry is None else dict(clip=geometry.box_counts,self=geometry.self_counts,
    ground=geometry.ground_counts,clip_self=geometry.rigid_counts,checked=geometry.checked_counts)
history={}
graph=None
if a.cuda_graph:
    history={name:[int(buffer.numpy()[0])] for name,buffer in buffers.items()}
    # Same even-substep state-buffer pattern as Newton's official cloth example.
    # Each graph replay writes a fresh per-frame ring; CPU concatenation preserves
    # one independent geometry record for every physics state, including reset.
    with wp.ScopedCapture() as capture:
        for buffer in buffers.values():buffer.zero_()
        for substep in range(a.substeps):advance_substep(substep+1)
    graph=capture.graph

for frame in range(round(a.seconds*60)):
    if graph:
        wp.capture_launch(graph)
        physics_step+=a.substeps
        for name,buffer in buffers.items():history[name].extend(buffer.numpy()[1:a.substeps+1].tolist())
    for _ in range(0 if graph else a.substeps):
        advance_substep(physics_step+1)
        physics_step+=1
        if geometry and a.stop_on_geometry_error:
            failed_box=int(geometry.box_counts.numpy()[physics_step])
            failed_self=int(geometry.self_counts.numpy()[physics_step])
            failed_ground=int(geometry.ground_counts.numpy()[physics_step])
            failed_rigid=int(geometry.rigid_counts.numpy()[physics_step])
            if failed_box or failed_self or failed_ground or failed_rigid:
                np.savez_compressed(a.output/'failure-state.npz',q=s0.particle_q.numpy(),
                    bodies=s0.body_q.numpy(),faces=f,box_flags=geometry.box_flags.numpy(),
                    self_flags=geometry.self_flags.numpy(),rigid_flags=geometry.rigid_flags.numpy())
                (a.output/'failure.json').write_text(json.dumps({'physics_step':physics_step,
                    'time_s':physics_step*dt,'box_intrusions':failed_box,'self_crossings':failed_self,
                    'ground_intrusions':failed_ground,'clip_self_intrusions':failed_rigid},indent=2))
                raise RuntimeError('Rejected: first geometry violation saved in failure-state.npz')
    measured=s0.particle_q.numpy().astype(float)
    gap=layer_gap(measured,entity)
    trace.append({"step":frame+1,"finite":bool(np.isfinite(measured).all()),
        "gap_m":float(gap) if np.isfinite(gap) else None,
        "max_speed_m_s":float(np.linalg.norm(s0.particle_qd.numpy(),axis=1).max())})
    if a.clip:
        from scipy.spatial.transform import Rotation
        body=s0.body_q.numpy()
        trace[-1]['finite']=bool(trace[-1]['finite'] and np.isfinite(body).all())
        jaw=(Rotation.from_quat(body[clip_root,3:]).inv()*Rotation.from_quat(body[clip_upper,3:])).as_rotvec()[1]
        n=min(int(contacts.soft_contact_count.numpy()[0]),contacts.soft_contact_max)
        ids=contacts.soft_contact_shape.numpy()[:n]
        forces=observables.contact_f.numpy()[contacts.rigid_contact_max:contacts.rigid_contact_max+n,:3]
        shape_body=m.shape_body.numpy()
        sums={str(i):float(np.linalg.norm(forces[shape_body[ids]==i],axis=1).sum()) for i in (clip_root,clip_upper)}
        nets={str(i):float(np.linalg.norm(forces[shape_body[ids]==i].sum(axis=0))) for i in (clip_root,clip_upper)}
        trace[-1].update(clip_root_pose=body[clip_root].tolist(),jaw_angle_rad=float(jaw),
            clip_soft_force_norm_sum_N=sums,clip_soft_net_force_N=nets,soft_contact_count=n)
        indices=contacts.soft_contact_indices.numpy()[:n]
        weights=contacts.soft_contact_barycentric.numpy()[:n]
        contact_points=(measured[np.maximum(indices,0)]*weights[:,:,None]).sum(axis=1)
        contact_local=Rotation.from_quat(body[clip_root,3:]).inv().apply(contact_points-body[clip_root,:3])
        normals=contacts.soft_contact_normal.numpy()[:n]
        shape_local=m.shape_transform.numpy()
        shape_reports={}
        for sid in np.flatnonzero(shape_body>=0):
            mask=(ids==sid)&(np.linalg.norm(forces,axis=1)>1e-5)
            in_fold=np.any((indices>=0)&(q[np.maximum(indices,0),0]>=.16),axis=1)
            raw_active=mask&in_fold
            active=raw_active.copy()
            if sid in jaw_groups[0] or sid in jaw_groups[1]:
                from check_native_clip import closing_face_mask
                sign=1. if sid in jaw_groups[0] else -1.
                active&=closing_face_mask(normals,body[shape_body[sid],3:],shape_local[sid,3:],sign)
            shape_reports[str(sid)]={"active_fold_contacts":int(active.sum()),
                "raw_active_fold_contacts":int(raw_active.sum()),
                "force_norm_sum_N":float(np.linalg.norm(forces[active],axis=1).sum()),
                "fold_contact_y_span_m":float(np.ptp(contact_local[active,1])) if active.any() else 0.,
                "fold_contact_y_min_m":float(contact_local[active,1].min()) if active.any() else None,
                "fold_contact_y_max_m":float(contact_local[active,1].max()) if active.any() else None}
        trace[-1]["shape_fold_contacts"]=shape_reports
        fold_local=Rotation.from_quat(body[clip_root,3:]).inv().apply(measured[fold_ids]-body[clip_root,:3])
        trace[-1]["fold_centroid_clip_local_m"]=fold_local.mean(axis=0).tolist()
    if frame%a.record_every==0 or frame==round(a.seconds*60)-1:
        snapshots[f"q_{frame+1:05d}"]=measured.copy()
        if a.clip:snapshots[f"b_{frame+1:05d}"]=body.copy()
    if frame%30==0:
        if geometry:
            trace[-1]["geometry_violations_so_far"]={name:int(np.count_nonzero(history[name] if graph else buffer.numpy()[:physics_step+1]))
                for name,buffer in buffers.items() if name!='checked'}
            (a.output/"progress.json").write_text(json.dumps(trace[-1],indent=2))
        print("PROGRESS",frame+1,trace[-1],flush=True)
rest_final=m.edge_rest_angle.numpy()
crease_delta=np.abs(np.arctan2(np.sin(rest_final-rest_initial),np.cos(rest_final-rest_initial)))[rows]
result={"residual_rest_angle_max_rad":float(crease_delta.max()),"residual_rest_angle_over_0_05_count":int((crease_delta>.05).sum()),"device":a.device,"self_contact":bool(a.self_contact),"initial_fold_imposed":a.folded,"material_profile":a.material_profile,"coefficients":coeff,"steps":len(trace),
  "wall_seconds":time.monotonic()-started,"physical_thickness_m":80e-6,
  "contact_radius_m":.0001,"mesh_cells":[a.nx,a.ny],"crease_hinges":len(rows),
  "all_finite":all(x["finite"] for x in trace),"mouth":mouth_opening(measured,entity),
  "clip":a.clip,"limitation":"homogeneous equivalent and uniform hinge width are assumptions; optional clip starts open around fold; no insertion, air, food or robot"}
result.update(substeps=a.substeps,physics_dt_s=dt,clip_offset_m=a.clip_offset,record_every=a.record_every,
    spring_damping=a.spring_damping,soft_contact_stiffness=a.contact_stiffness,
    clip_contact_margin_m=a.clip_contact_margin,rigid_contact_stiffness=a.rigid_contact_stiffness,iterations=iterations,
    collision_refresh=a.collision_refresh,rigid_soft_dat=a.rigid_soft_dat)
result.update(initial_open_angle=a.initial_open_angle,open_limit=a.open_limit,mouth_drop_m=a.mouth_drop)
result['ground_contact_margin_m']=a.ground_contact_margin
result['cuda_graph']=a.cuda_graph
result['compliant_jaw']=a.compliant_jaw
result['opposed_contact_faces']=bool(a.clip)
result['spring_stiffness']=a.spring_stiffness
result['topology_filter']=a.topology_filter
result.update(pad_stiffness=a.pad_stiffness,pad_target=a.pad_target,pad_damping=a.pad_damping)
if geometry:
    values={name:np.asarray(history[name]) if graph else buffer.numpy() for name,buffer in buffers.items()}
    box_counts,self_counts=values['clip'],values['self']
    ground_counts,rigid_counts=values['ground'],values['clip_self']
    np.savez_compressed(a.output/"geometry-counts.npz",clip_intrusions=box_counts,self_crossings=self_counts,
        ground_intrusions=ground_counts,clip_self_intrusions=rigid_counts,checks_per_state=values['checked'])
    result["geometry"]={"states_checked":int(np.count_nonzero(values['checked'])),
        "checks_per_state_min":int(values['checked'].min()),"checks_per_state_max":int(values['checked'].max()),
        "clip_intrusion_states":int(np.count_nonzero(box_counts)),
        "clip_intrusion_max":int(box_counts.max()),"self_crossing_states":int(np.count_nonzero(self_counts)),
        "self_crossing_max":int(self_counts.max()),"ground_intrusion_states":int(np.count_nonzero(ground_counts)),
        "ground_intrusion_max":int(ground_counts.max()),
        "clip_self_intrusion_states":int(np.count_nonzero(rigid_counts)),"clip_self_intrusion_max":int(rigid_counts.max()),
        "scope":"triangle/box interiors and z=0 plane with 0.1um tolerance; strict transverse self crossings; excludes coplanar overlap and inter-step motion"}
(a.output/"result.json").write_text(json.dumps(result,indent=2))
(a.output/"trace.json").write_text(json.dumps(trace))
np.savez_compressed(a.output/"pouch-snapshots.npz",faces=f,edges=edges,rest_initial=rest_initial,rest_final=rest_final,**snapshots)
print("RESULT",json.dumps(result),flush=True)
