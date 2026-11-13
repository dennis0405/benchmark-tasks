"""RoboSimStudio MuJoCo/VBD coupling diagnostic, not a robot insertion demo."""
import json,sys,time
from pathlib import Path
from dataclasses import dataclass,field
from typing import Any
import numpy as np
import warp as wp
import newton
import robosimstudio as rss
from scipy.spatial.transform import Rotation
from robosimstudio_materials.paper import EntityPaper,PaperSpec,PAPER
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pouch_geometry import open_pouch
from bag_surface_mesh import folded_positions
from film_material import FilmMaterial
from check_native_clip import inspect_state
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
frames=int(sys.argv[2]) if len(sys.argv)>2 else 240
NX,NY=80,64
coeff=FilmMaterial().coefficients(.0025)
material=rss.replace(PAPER,zero_rest_angle=False,thickness=80e-6,areal_density=coeff["areal_density"],
  tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],edge_ke=coeff["edge_ke"],
  particle_radius=.0001,yield_angle=coeff["yield_angle"],plastic_recovery=.1)
@dataclass(frozen=True,kw_only=True)
class FilmSpec(PaperSpec):
    entity:str="snack_film_pouch"
@rss.register_entity("snack_film_pouch")
class FilmEntity(EntityPaper):
    domain="paper"
    def build(self,builder,episode):
        q,f,self.grid=open_pouch(nx=NX,ny=NY,mouth_half_gap=.0003,foldable_mouth=True)
        pose=self.placed((*self.spec.pose,*self.spec.quat))
        self.surface_count=len(q);self.lumping="mesh"
        builder.add_cloth_mesh(pos=wp.vec3(*map(float,pose[:3])),rot=wp.quat(*map(float,pose[3:])),
          scale=1.,vel=wp.vec3(0),vertices=[wp.vec3(*map(float,x)) for x in q],
          indices=f.ravel().tolist(),density=material.areal_density,tri_ke=material.tri_ke,
          tri_ka=material.tri_ka,tri_kd=material.tri_kd,edge_ke=material.edge_ke,
          edge_kd=material.edge_kd,particle_radius=material.particle_radius,label=self.name)
    def bind(self,model):
        super().bind(model)
        stepping=self.stepping_model if self.stepping_model is not None else model
        edges=stepping.edge_indices.numpy()[self.rows]
        points=stepping.particle_q.numpy()
        valid=np.all(edges>=0,axis=1)
        mid=points[np.maximum(edges,0)].mean(axis=1)
        valid &= mid[:,0]>=points[:,0].min()+.14
        self.rows=self.rows[valid]
        self._rows=wp.array(self.rows,dtype=wp.int32,device=stepping.device)
@dataclass(frozen=True,kw_only=True)
class ClipSpec:
    entity:str="snack_spring_clip"
    name:str="clip"
    sim:Any=field(default_factory=lambda:rss.SimRequirements(domain="articulated",
      contacts=rss.contacts_with("surface","objects","paper","cloth","deformables")))
@rss.register_entity("snack_spring_clip")
class ClipEntity(rss.Entity):
    domain="articulated"
    def build(self,builder,episode):
        body_start,shape_start=builder.body_count,builder.shape_count
        pose=self.placed((.184,0,.0252,0,0,1,0))
        builder.add_usd(str(Path(__file__).resolve().parents[1]/"assets/clip_v5/clip_v5.usda"),
          floating=True,xform=wp.transform(wp.vec3(*map(float,pose[:3])),wp.quat(*map(float,pose[3:]))),
          override_root_xform=True)
        h=next(i for i,n in enumerate(builder.joint_label) if n.endswith("clip_hinge"))
        qi,di=builder.joint_q_start[h],builder.joint_qd_start[h]
        builder.joint_q[qi]=-.55;builder.joint_target_q[qi]=0.
        builder.joint_target_ke[di]=.13;builder.joint_target_kd[di]=.003
        builder.joint_target_mode[di]=newton.JointTargetMode.POSITION
        self.bodies=tuple(range(body_start,builder.body_count))
        self.shapes=tuple(range(shape_start,builder.shape_count))
    def handle(self,model):
        from robosimstudio.scene.handles import EntityHandle
        return EntityHandle(name=self.name,body=self.bodies[0],bodies=self.bodies,shapes=self.shapes)
    def on_scene_ready(self,pipeline):
        self.reset(pipeline,None)
    def reset(self,pipeline,episode,*,world=0):
        for state in (pipeline.state_0,pipeline.state_1):
            if state.joint_q is not None:state.joint_q.assign(self.model.joint_q)
            if state.joint_qd is not None:state.joint_qd.zero_()
            state.body_qd.zero_()
            newton.eval_fk(self.model,self.model.joint_q,self.model.joint_qd,state)
physics=rss.replace(rss.GENERIC.with_backend("articulated",rss.backend("mujoco",iterations=100,ls_iterations=50)).with_backend("paper",rss.backend("vbd",iterations=20,
    particle_self_contact_radius=.0001,particle_self_contact_margin=.001,
    rigid_body_particle_contact_buffer_size=262144)),
    collision=rss.CollisionSpec(full_surface_contact=True,soft_contact_margin=.0005,rigid_gap=.0005))
scene=rss.Scene(cell="bench_0728",physics=physics,name="snack_clip_coupled")
scene.add(FilmSpec(name="film",pose=(0,0,.025),material=material,settle_frames=0))
scene.add(ClipSpec())
sim=scene.build(device="cuda:0")
print("SOLVERS",type(sim.pipeline.program.solver_for("film")).__name__,
      type(sim.pipeline.program.solver_for("clip")).__name__,"coupled",sim.pipeline.is_coupled,flush=True)
entity=sim.scene.by_name["film"];ids=np.asarray(entity.particles,dtype=int)
q,f,grid=open_pouch(nx=NX,ny=NY,mouth_half_gap=.0003,foldable_mouth=True)
before=sim.state.particle_q.numpy();offset=np.median(before[ids]-q,axis=0)
initial=folded_positions(q+[0,0,.019],radius=.0018)-[0,0,.019]+offset
for state in (sim.pipeline.state_0,sim.pipeline.state_1):
    points=state.particle_q.numpy();points[ids]=initial;state.particle_q.assign(points)
sim.pipeline.invalidate_graph()
model=sim.model
for state in (sim.pipeline.state_0,sim.pipeline.state_1):
    if state.joint_q is not None:state.joint_q.assign(model.joint_q)
    if state.joint_qd is not None:state.joint_qd.zero_()
    state.body_qd.zero_();state.particle_qd.zero_()
    newton.eval_fk(model,model.joint_q,model.joint_qd,state)
sim.pipeline.reset_solvers(flags=0)
sim.pipeline.reset_finite_flags()
sim.pipeline.invalidate_graph()
root=next(i for i,n in enumerate(model.body_label) if n.endswith("/Geometry/clip"))
upper=next(i for i,n in enumerate(model.body_label) if n.endswith("/Geometry/clip/upper"))
shape={"body_labels":list(model.body_label),"shape_body":model.shape_body.numpy().tolist(),
  "shape_local":model.shape_transform.numpy().tolist(),"shape_scale":model.shape_scale.numpy().tolist(),
  "root":root,"upper":upper}
# Only the four clip boxes are relevant to the clip intrusion diagnostic.
shape["shape_body"]=[owner if owner in (root,upper) else -1 for owner in shape["shape_body"]]
(out/"clip-shapes.json").write_text(json.dumps(shape,indent=2))
preflight=inspect_state(initial,f,sim.state.body_q.numpy(),shape)
print("PREFLIGHT",preflight,flush=True)
(out/"preflight.json").write_text(json.dumps(preflight,indent=2))
if any(x["penetrating_triangles"] for x in preflight):raise RuntimeError("initial intrusion")
snapshots={"q_00000":initial.copy(),"b_00000":sim.state.body_q.numpy().copy()}
trace=[];started=time.monotonic()
for frame in range(frames):
    sim.run(1).sync();points=sim.state.particle_q.numpy()[ids];body=sim.state.body_q.numpy()
    if not sim.state_is_finite() or min(np.linalg.norm(body[root,3:]),np.linalg.norm(body[upper,3:]))<1e-6:
        trace.append({"step":frame+1,"finite":False,"failure":"nonfinite state or invalid rigid quaternion"})
        print("INVALID",trace[-1],flush=True)
        break
    jaw=(Rotation.from_quat(body[root,3:]).inv()*Rotation.from_quat(body[upper,3:])).as_rotvec()[1]
    trace.append({"step":frame+1,"finite":bool(np.isfinite(points).all() and np.isfinite(body).all()),
      "jaw_angle_rad":float(jaw),"clip_root_pose":body[root].tolist(),
      "max_speed_m_s":float(np.linalg.norm(sim.state.particle_qd.numpy()[ids],axis=1).max())})
    if frame%15==0 or frame==frames-1:
        snapshots[f"q_{frame+1:05d}"]=points.copy();snapshots[f"b_{frame+1:05d}"]=body.copy()
    if frame%30==0:print("PROGRESS",trace[-1],flush=True)
result={"coupled":bool(sim.pipeline.is_coupled),"steps":len(trace),"wall_seconds":time.monotonic()-started,
  "all_finite":all(x["finite"] for x in trace),"coefficients":coeff,
  "limitation":"fold imposed at reset; free spring clip starts open around mouth; no insertion, robot, food or air"}
(out/"result.json").write_text(json.dumps(result,indent=2));(out/"trace.json").write_text(json.dumps(trace))
np.savez_compressed(out/"pouch-snapshots.npz",faces=f,**snapshots)
print("RESULT",result,flush=True)
