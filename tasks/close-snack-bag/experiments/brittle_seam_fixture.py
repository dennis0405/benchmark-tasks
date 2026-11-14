"""Brittle seam component fixture, not a snack-task episode or calibration.

Two pre-existing rigid fragments share a compliant rod seam. A test couple
loads it; exceeding a measured bend angle disables the seam irreversibly.
The fixture has zero gravity and external viscous drag. No poses are driven.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import newton
from newton.solvers import SolverVBD
import warp as wp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--torque',type=float,default=.01)
    p.add_argument('--steps',type=int,default=600)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    builder=newton.ModelBuilder();bodies=[]
    cfg=newton.ModelBuilder.ShapeConfig(density=524.,ke=1e5,kd=10.,mu=.5,margin=0.)
    for x in (-.012,.012):
        body=builder.add_link(xform=wp.transform(wp.vec3(x,0,.1),wp.quat_identity()))
        builder.add_shape_box(body,hx=.012,hy=.018,hz=.0012,cfg=cfg)
        bodies.append(body)
    tangent=wp.quat_from_axis_angle(wp.vec3(0,1,0),np.pi/2)
    joint=builder.add_joint_rod(*bodies,
        parent_xform=wp.transform(wp.vec3(.012,0,0),tangent),
        child_xform=wp.transform(wp.vec3(-.012,0,0),tangent),
        stretch_stiffness=1e4,shear_stiffness=1e4,bend_stiffness=.05,twist_stiffness=.05,
        stretch_damping=2.,shear_damping=2.,bend_damping=.0001,twist_damping=.0001,
        collision_filter_parent=False,label='brittle_seam')
    builder.add_articulation([joint]);builder.color()
    model=builder.finalize(device='cuda:0');model.gravity.zero_()
    solver=SolverVBD(model,iterations=30,particle_enable_self_contact=False,rigid_compliant_alm=True)
    collision=newton.CollisionPipeline(model);contacts=collision.contacts();control=model.control()
    current,following=model.state(),model.state();dt=1/2400.;broken=False;event=None
    trace=[];snapshots={}
    for step in range(a.steps+1):
        poses=current.body_q.numpy();rates=current.body_qd.numpy()
        relative=Rotation.from_quat(poses[0,3:]).inv()*Rotation.from_quat(poses[1,3:])
        angle=float(relative.magnitude())
        if not broken and angle>.12:
            enabled=model.joint_enabled.numpy().copy();enabled[joint]=False;model.joint_enabled.assign(enabled)
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            broken=True;event=dict(step=step,time_s=step*dt,angle_rad=angle)
        trace.append(dict(step=step,time_s=step*dt,angle_rad=angle,broken=broken,
            finite=bool(np.isfinite(poses).all() and np.isfinite(rates).all())))
        snapshots[f'b_{step:05d}']=poses.copy()
        if step==a.steps:break
        current.clear_forces()
        forces=np.zeros_like(current.body_f.numpy())
        # Instrumented test loading only, absent from any task controller.
        forces[:,3:]=-.00002*rates[:,3:]
        if not broken and step<240:
            forces[0,4]-=a.torque;forces[1,4]+=a.torque
        current.body_f.assign(forces)
        collision.collide(current,contacts);solver.step(current,following,control,contacts,dt)
        current,following=following,current
    result=dict(scope=__doc__,torque_Nm=a.torque,threshold_rad=.12,event=event,
        broken=broken,final_enabled=bool(model.joint_enabled.numpy()[joint]),
        all_finite=all(t['finite'] for t in trace),max_bend_rad=max(t['angle_rad'] for t in trace),
        final_bend_rad=trace[-1]['angle_rad'],joint_coordinates=model.joint_coord_count,
        mass_kg=model.body_mass.numpy().tolist(),calibrated=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    (a.output/'trace.json').write_text(json.dumps(trace))
    np.savez_compressed(a.output/'states.npz',**snapshots)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
