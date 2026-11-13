"""Fixed-base passive clip closure on four separate film layers.

This isolates the dynamic upper jaw / explicit film reaction. No YAM, pouch,
free-tool retention, air or task score is claimed by this fixture.
"""
import argparse
from dataclasses import dataclass,field
import json
from pathlib import Path
import sys,time
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
    sim:object=field(default_factory=lambda:rss.SimRequirements(domain="articulated",
        contacts=rss.contacts_with("paper","objects")))


@rss.register_entity("snack_closure_fixed_clip")
class FixedClip(rss.Entity):
    domain="articulated"
    def build(self,builder,episode):
        del episode
        start=builder.body_count
        frame=self.placed((0,0,.100,0,0,0,1))
        builder.add_mjcf(self.spec.xml,floating=False,override_root_xform=True,
            xform=wp.transform(wp.vec3(*frame[:3]),wp.quat(*frame[3:])))
        self.bodies=tuple(range(start,builder.body_count))
        for body in self.bodies:builder.body_label[body]="snack_clip/"+builder.body_label[body]


def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    p.add_argument("--ke",type=float,default=1e6);p.add_argument("--frames",type=int,default=100)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=a.output/"sources";sources.mkdir()
    root=Path(__file__).resolve().parents[1]
    for source in (Path(__file__),root/"deformable_scene.py",root/"pouch_geometry.py",
                   Path(__file__).with_name("film_material.py"),Path(__file__).with_name("tool_feedback.py"),
                   Path(__file__).with_name("substep_geometry.py"),Path(__file__).with_name("gpu_contact_checks.py"),
                   Path(__file__).with_name("assets")/"clip_thin_lower.xml"):
        shutil.copy2(source,sources/source.name)
    (a.output/"arguments.json").write_text(json.dumps(dict(vars(a),hz=20,fixture=True),default=str,indent=2))
    # Equivalent joint-coordinate shift: initial -0.32 rad opening, passive
    # spring equilibrium at closed. There is no object command during rollout.
    tree=ET.parse(Path(__file__).with_name("assets")/"clip_thin_lower.xml")
    body=tree.getroot().find("worldbody/body");body.remove(body.find("freejoint"))
    upper=body.find("body");upper.set("quat",".9872272834 0 -.1593182066 0")
    joint=upper.find("joint");joint.set("springref",".32");joint.set("range","-.08 .326")
    asset=a.output/"fixed-open-clip.xml";tree.write(asset)
    coeff=FilmMaterial(young=1e9,recovery_per_second=0).coefficients(.005)
    material=rss.replace(PAPER,zero_rest_angle=False,thickness=80e-6,areal_density=.08,
        tri_ke=coeff["tri_ke"],tri_ka=coeff["tri_ka"],edge_ke=coeff["edge_ke"],
        particle_radius=.0003,yield_angle=.5,plastic_recovery=0.,contact_ke=1e4,friction=.5)
    physics=rss.replace(rss.GENERIC,control_hz=20.,substeps=120,
        coupling=rss.coupler("one_way",contact_forces=True,slide_feedback=False),
        collision=rss.CollisionSpec(full_surface_contact=True))
    physics=physics.with_backend("*",rss.backend("vbd",iterations=30,rigid_compliant_alm=True,
        particle_self_contact_radius=.0005,particle_self_contact_margin=.002,
        particle_topological_contact_filter_threshold=0,rigid_body_particle_contact_buffer_size=16384,
        particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128))
    scene=rss.Scene(cell="bench_0728",physics=physics,name="fixed_closure_fixture")
    scene.add(rss.slab(name="support",half=(.0325,.10,.0495),at=(.0475,0,.0495)))
    scene.add(PouchSpec(entity="snack_closure_layers",name="layers",pose=(0,0,.1),
                        quat=(0,0,0,1),cells=(5,32),size=(.025,.16),material=material,settle_frames=0))
    scene.add(FixedClipSpec(xml=str(asset)))
    started=time.monotonic();sim=scene.build(device="cuda:0",control_mode="joint_position")
    for state in (sim.pipeline.state_0,sim.pipeline.state_1):
        state.particle_q.assign(sim.model.particle_q);state.particle_qd.zero_()
        state.joint_q.assign(sim.model.joint_q);state.joint_qd.zero_();state.body_qd.zero_()
        newton.eval_fk(sim.model,sim.model.joint_q,sim.model.joint_qd,state)
    sim.pipeline.reset_solvers(flags=0);sim.pipeline.reset_finite_flags()
    owners=sim.model.shape_body.numpy()
    bodies=sim.scene.by_name["clip"].bodies
    ke=sim.model.shape_material_ke.numpy().copy();ke[np.isin(owners,bodies)]=a.ke
    sim.model.shape_material_ke.assign(ke)
    feedback=ToolFeedback(sim.pipeline.program.solver)
    sim.pipeline.hooks.add("pre_solve",feedback.before_solve,name="fixture-feedback",cadence=1)
    entity=sim.scene.by_name["layers"];watcher=SubstepGeometry(sim,entity)
    sim.pipeline.hooks.add("post_solve",watcher.check,name="fixture-geometry",cadence=1)
    sim.pipeline.invalidate_graph();export_render_geometry(sim,a.output)
    data=dict(faces=entity.faces,grid=entity.grid);trace=[]
    for step in range(a.frames+1):
        if step:sim.run(1).sync()
        q=sim.state.particle_q.numpy()[entity.particles];b=sim.state.body_q.numpy()
        angle=float((Rotation.from_quat(b[bodies[0],3:]).inv()*Rotation.from_quat(b[bodies[1],3:])).as_rotvec()[1])
        force=sim.pipeline.program.solver.shape_contact_forces()
        row=dict(step=step,phase="fixed_base_closure",finite=bool(sim.state_is_finite()),
                 clip_root_pose=b[bodies[0]].tolist(),clip_angle_rad=angle,
                 film_force_N=[force[owners==body].sum(0).tolist() for body in bodies],geometry=watcher.report())
        trace.append(row)
        if step%5==0:
            data[f"q_{step:05d}"]=q;data[f"b_{step:05d}"]=b
        if step%10==0:print(row,flush=True)
        if not row["finite"] or row["geometry"]["transverse_edge_crossings"] or row["geometry"]["film_box_intersections"]:break
    np.savez_compressed(a.output/"states.npz",**data)
    (a.output/"trace.json").write_text(json.dumps(trace))
    result=dict(scope=__doc__,ke=a.ke,frames=trace[-1]["step"],wall_s=time.monotonic()-started,
                all_finite=all(t["finite"] for t in trace),geometry=watcher.report(),final_angle_rad=trace[-1]["clip_angle_rad"])
    (a.output/"result.json").write_text(json.dumps(result,indent=2));print("RESULT",result,flush=True)
    sim.close()


if __name__=="__main__":main()
