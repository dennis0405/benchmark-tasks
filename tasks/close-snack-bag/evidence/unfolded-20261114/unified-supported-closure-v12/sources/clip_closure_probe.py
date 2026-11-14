"""Passive clip closure on four separate film layers.

Default: fixed base. --free-root adds a passive heel stand instead of fixing
the base. No YAM, pouch, unsupported retention, air or task score is claimed.
"""
import argparse
from dataclasses import dataclass,field
import json
from pathlib import Path
import sys,time,inspect
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import robosimstudio as rss
from deformable_scene import PouchEntity,PouchSpec,export_render_geometry
from robosimstudio_materials.paper import PAPER
from film_material import FilmMaterial
from tool_feedback import ToolFeedback
from substep_geometry import SubstepGeometry


@rss.register_entity("snack_closure_layers")
class Layers(PouchEntity):
    def surface(self):
        nx,ny=self.spec.cells
        self.grid=np.arange(4*(nx+1)*(ny+1)).reshape(4,nx+1,ny+1)
        q=np.array([[.015+i*.025/nx,-.08+j*.16/ny,.0006+layer*.0006]
                    for layer in range(4) for i in range(nx+1) for j in range(ny+1)])
        f=[]
        for g in self.grid:
            for i in range(nx):
                for j in range(ny):
                    a,b,c,d=g[i,j],g[i+1,j],g[i+1,j+1],g[i,j+1]
                    f.extend(((a,b,c),(a,c,d)))
        return q,np.asarray(f,np.int32)


@dataclass(frozen=True,kw_only=True)
class FixedClipSpec:
    entity:str="snack_closure_fixed_clip"
    name:str="clip"
    xml:str
    free_root:bool=False
    sim:object=field(default_factory=lambda:rss.SimRequirements(domain="articulated",
        contacts=rss.contacts_with("paper","objects")))


@rss.register_entity("snack_closure_fixed_clip")
class FixedClip(rss.Entity):
    domain="articulated"
    def build(self,builder,episode):
        del episode
        start=builder.body_count
        frame=self.placed((0,0,.100,0,0,0,1))
        builder.add_mjcf(self.spec.xml,floating=self.spec.free_root,override_root_xform=True,
            xform=wp.transform(wp.vec3(*frame[:3]),wp.quat(*frame[3:])))
        self.bodies=tuple(range(start,builder.body_count))
        for body in self.bodies:builder.body_label[body]="snack_clip/"+builder.body_label[body]


def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    p.add_argument("--ke",type=float,default=1e6);p.add_argument("--frames",type=int,default=100)
    p.add_argument("--free-root",action="store_true")
    p.add_argument("--no-feedback",action="store_true",help="Negative control only; no film wrench on the tool")
    p.add_argument("--record-forces",action="store_true")
    p.add_argument("--unified",action="store_true",help="Solve rigid tool and film together with Newton VBD")
    p.add_argument("--physics-hz",type=int,default=2400)
    p.add_argument("--particle-buffer",type=int,default=16384)
    p.add_argument("--clip-style",choices=("thin_lower","thin_compact"),default="thin_lower")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.unified and a.no_feedback:p.error("unified uses intrinsic two-way contact; no-feedback is inapplicable")
    if a.physics_hz<20 or a.physics_hz%20:p.error("physics-hz must be a positive multiple of 20")
    sources=a.output/"sources";sources.mkdir()
    root=Path(__file__).resolve().parents[1]
    asset_source=Path(__file__).with_name("assets")/f"clip_{a.clip_style}.xml"
    for source in (Path(__file__),root/"deformable_scene.py",root/"pouch_geometry.py",
                   Path(__file__).with_name("film_material.py"),Path(__file__).with_name("tool_feedback.py"),
                   Path(__file__).with_name("substep_geometry.py"),Path(__file__).with_name("gpu_contact_checks.py"),
                   Path(__file__).with_name("round_contact_checks.py"),
                   asset_source):
        shutil.copy2(source,sources/source.name)
    (a.output/"arguments.json").write_text(json.dumps(dict(vars(a),hz=20,fixture=True),default=str,indent=2))
    # Equivalent joint-coordinate shift: initial -0.32 rad opening, passive
    # spring equilibrium at closed. There is no object command during rollout.
    tree=ET.parse(asset_source)
    body=tree.getroot().find("worldbody/body")
    if not a.free_root:body.remove(body.find("freejoint"))
    upper=body.find("body");upper.set("quat",".9872272834 0 -.1593182066 0")
    joint=upper.find("joint");joint.set("springref",".32");joint.set("range","-.08 .326")
    stiffness=float(joint.get("stiffness","0"));damping=float(joint.get("damping","0"))
    if a.unified:
        # Explicitly author the same passive law in Newton: MuJoCo's
        # imported passive spring is not a generic VBD actuator.
        joint.set("stiffness","0");joint.set("damping","0")
        source=Path(__file__).with_name("passive_clip_spring.py")
        shutil.copy2(source,sources/source.name)
    asset=a.output/("free-open-clip.xml" if a.free_root else "fixed-open-clip.xml");tree.write(asset)
    coeff=FilmMaterial(young=1e9,recovery_per_second=0).coefficients(.005)
    material=rss.replace(PAPER,zero_rest_angle=False,thickness=80e-6,areal_density=.08,
        tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],edge_ke=coeff["edge_ke"],
        particle_radius=.0003,yield_angle=.5,plastic_recovery=0.,contact_ke=1e4,friction=.5)
    physics=rss.replace(rss.GENERIC,control_hz=20.,substeps=a.physics_hz//20,
        coupling=(rss.coupler("proxy",iterations=2,proxy_joints=True) if a.unified else
                  rss.coupler("one_way",contact_forces=True,slide_feedback=False)),
        collision=rss.CollisionSpec(full_surface_contact=True))
    backend=rss.backend("vbd",iterations=30,rigid_compliant_alm=True,
        particle_self_contact_radius=.0005,particle_self_contact_margin=.002,
        particle_topological_contact_filter_threshold=0,rigid_body_contact_buffer_size=2048,
        rigid_body_particle_contact_buffer_size=a.particle_buffer,
        particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128)
    physics=physics.with_backend("*",backend)
    if a.unified:physics=physics.with_backend("articulated",backend)
    scene=rss.Scene(cell="bench_0728",physics=physics,name="fixed_closure_fixture")
    scene.add(rss.slab(name="support",half=(.0325,.10,.0495),at=(.0475,0,.0495)))
    if a.free_root:
        scene.add(rss.slab(name="heel_support",half=(.040,.05,.0285),at=(-.07,0,.0285)))
    scene.add(PouchSpec(entity="snack_closure_layers",name="layers",pose=(0,0,.1),
                        quat=(0,0,0,1),cells=(5,32),size=(.025,.16),material=material,settle_frames=0,
                        structural_fk=a.unified))
    scene.add(FixedClipSpec(xml=str(asset),free_root=a.free_root))
    started=time.monotonic();sim=scene.build(device="cuda:0",control_mode="joint_position")
    build_s=time.monotonic()-started
    runtime=dict(pipeline_type=type(sim.pipeline).__module__+"."+type(sim.pipeline).__name__,
        pipeline_source=inspect.getfile(type(sim.pipeline)),program_source=inspect.getfile(type(sim.pipeline.program)),
        program_fields=list(vars(sim.pipeline.program)),use_cuda_graph=sim.pipeline.use_cuda_graph,
        solver_type=type(sim.pipeline.program.solver).__name__,build_s=build_s)
    (a.output/"runtime.json").write_text(json.dumps(runtime,indent=2));print("RUNTIME",runtime,flush=True)
    for state in (sim.pipeline.state_0,sim.pipeline.state_1):
        state.particle_q.assign(sim.model.particle_q);state.particle_qd.zero_()
        state.joint_q.assign(sim.model.joint_q);state.joint_qd.zero_();state.body_qd.zero_()
        newton.eval_fk(sim.model,sim.model.joint_q,sim.model.joint_qd,state)
    sim.pipeline.reset_solvers(flags=0);sim.pipeline.reset_finite_flags()
    owners=sim.model.shape_body.numpy()
    bodies=sim.scene.by_name["clip"].bodies
    solver=sim.pipeline.program.solver
    rigid=getattr(solver,"rigid",None)
    native_solref=rigid.mj_model.geom_solref.copy() if rigid is not None else None
    ke=sim.model.shape_material_ke.numpy().copy();ke[np.isin(owners,bodies)]=a.ke
    sim.model.shape_material_ke.assign(ke)
    # Deliberate film-contact comparison only: the native rigid solver was
    # already built, so its robot/tool/stand contact properties stay fixed.
    native_unchanged=np.array_equal(native_solref,rigid.mj_model.geom_solref) if rigid is not None else None
    if native_unchanged is False:raise RuntimeError("native rigid properties changed")
    if not a.no_feedback and not a.unified:
        feedback=ToolFeedback(sim.pipeline.program.solver)
        sim.pipeline.hooks.add("pre_solve",feedback.before_solve,name="fixture-feedback",cadence=1)
    spring=None
    if a.unified:
        from passive_clip_spring import PassiveClipSpring
        spring=PassiveClipSpring(sim,bodies[0],bodies[1],stiffness,damping)
        sim.pipeline.hooks.add("pre_solve",spring.before_solve,name="passive-clip-spring",cadence=1)
    if a.unified and a.record_forces:
        from unified_contact import install
        install(sim)
        for name in ("unified_contact.py","contact_average.py"):
            shutil.copy2(Path(__file__).with_name(name),sources/name)
    entity=sim.scene.by_name["layers"];watcher=SubstepGeometry(sim,entity)
    sim.pipeline.hooks.add("post_solve",watcher.check,name="fixture-geometry",cadence=1)
    sim.pipeline.invalidate_graph();export_render_geometry(sim,a.output)
    data=dict(faces=entity.faces,grid=entity.grid);trace=[];ticks=[]
    for step in range(a.frames+1):
        tick_start=time.monotonic()
        if step:sim.run(1).sync();ticks.append(time.monotonic()-tick_start)
        q=sim.state.particle_q.numpy()[entity.particles];b=sim.state.body_q.numpy()
        finite=bool(sim.state_is_finite()) and bool(np.isfinite(b).all()) and all(
            np.linalg.norm(b[body,3:])>.5 for body in bodies)
        angle=(float((Rotation.from_quat(b[bodies[0],3:]).inv()*
                      Rotation.from_quat(b[bodies[1],3:])).as_rotvec()[1]) if finite else None)
        reader=getattr(sim.pipeline.program.solver,"shape_contact_forces",None)
        force=reader() if reader is not None else None
        row=dict(step=step,phase="supported_free_closure" if a.free_root else "fixed_base_closure",finite=finite,
                 body_values_finite=bool(np.isfinite(b).all()),particle_values_finite=bool(np.isfinite(q).all()),
                 clip_root_pose=b[bodies[0]].tolist(),clip_angle_rad=angle,
                 film_force_N=[force[owners==body].sum(0).tolist() for body in bodies] if force is not None else None,
                 passive_torque_Nm=float(spring.observed.numpy()[0]) if spring else None,
                 tick_wall_s=ticks[-1] if ticks and step else 0.,
                 geometry=watcher.report())
        trace.append(row)
        if step%5==0 or not finite:
            data[f"q_{step:05d}"]=q;data[f"b_{step:05d}"]=b
            data[f"j_{step:05d}"]=sim.state.joint_q.numpy()
            data[f"d_{step:05d}"]=sim.state.joint_qd.numpy()
        if step%10==0 or step<=3:print(row,flush=True)
        if not row["finite"] or row["geometry"]["transverse_edge_crossings"] or row["geometry"]["film_box_intersections"] or row["geometry"]["film_round_intersections"]:break
    np.savez_compressed(a.output/"states.npz",**data)
    (a.output/"trace.json").write_text(json.dumps(trace))
    result=dict(scope=__doc__,ke=a.ke,free_root=a.free_root,clip_style=a.clip_style,
                unified=a.unified,force_reader_available=reader is not None,
                physics_hz=a.physics_hz,particle_buffer=a.particle_buffer,
                build_s=build_s,first_tick_s=ticks[0] if ticks else None,
                mean_later_tick_s=float(np.mean(ticks[1:])) if len(ticks)>1 else None,
                passive_spring=dict(stiffness=stiffness,damping=damping,adapter=bool(spring)),
                negative_control_no_feedback=a.no_feedback,
                native_rigid_solref_unchanged=native_unchanged,
                nonfinite_problem_report=str(sim.pipeline.program.problems()) if not trace[-1]["finite"] else None,
                frames=trace[-1]["step"],wall_s=time.monotonic()-started,
                all_finite=all(t["finite"] for t in trace),geometry=watcher.report(),final_angle_rad=trace[-1]["clip_angle_rad"])
    (a.output/"result.json").write_text(json.dumps(result,indent=2));print("RESULT",result,flush=True)
    sim.close()


if __name__=="__main__":main()
