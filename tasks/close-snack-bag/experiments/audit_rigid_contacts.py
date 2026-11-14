"""Reconstruct native MuJoCo rigid contacts at a recorded joint snapshot.

Offline diagnostic only: no recorded rollout is edited. Contact distances
and geometry metadata are inspected; forward-computed forces are not claimed
to reproduce the original control/solver caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import newton
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--step",type=int);p.add_argument("--allow-body-residual",action="store_true");a=p.parse_args()
    args=json.loads((a.run/"arguments.json").read_text())
    raw=(a.run/"states.npz").read_bytes()
    import io
    states=np.load(io.BytesIO(raw))
    step=a.step if a.step is not None else max(int(k[2:]) for k in states.files if k.startswith("b_"))
    sim=make_environment(ready_start=True,with_clip=True,rigid_only=True,rigid_cpu=True,
        control_hz=20,physics_hz=args.get("physics_hz",2400),workspace=args["workspace"],
        clip_style=args["clip_style"],side_access=args.get("mouth_support","full")=="narrow",
        support_extension=args.get("support_extension_mm",0.)/1000.,
        board_friction=args.get("board_friction",1.),finger_friction=args.get("finger_friction"),
        rebase_z=args.get("rebase_z",False),board_skin_mm=args.get("board_skin_mm",0.),
        tool_rack_style=args.get("tool_rack_style","wide"),tool_parking_shift_x=args.get("tool_parking_shift_x",0.))
    sim.state.joint_q.assign(states[f"j_{step:05d}"])
    sim.state.joint_qd.assign(states[f"d_{step:05d}"])
    newton.eval_fk(sim.model,sim.state.joint_q,sim.state.joint_qd,sim.state)
    position_error=float(abs(sim.state.body_q.numpy()[:,:3]-states[f"b_{step:05d}"][:,:3]).max())
    if position_error>1e-5 and not a.allow_body_residual:raise ValueError("offline rigid model FK differs from the recording")
    solver=sim.pipeline.program.solver
    solver=getattr(solver,"rigid",solver)
    solver._update_mjc_data(solver.mj_data,sim.model,sim.state)
    model,data=solver.mj_model,solver.mj_data
    mujoco.mj_forward(model,data)
    mapping=solver.mjc_geom_to_newton_shape.numpy()[0]
    contacts=[]
    for i in range(data.ncon):
        contact=data.contact[i];geoms=[int(x) for x in contact.geom]
        shapes=[int(mapping[g]) for g in geoms]
        labels=[str(sim.model.shape_label[s]) if s>=0 else mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
                for s,g in zip(shapes,geoms)]
        owners=[int(sim.model.shape_body.numpy()[s]) if s>=0 else None for s in shapes]
        contacts.append(dict(geoms=geoms,shapes=shapes,owners=owners,labels=labels,
            distance_m=float(contact.dist),position_m=contact.pos.tolist(),
            constraint_address=int(contact.efc_address),included_margin_m=float(contact.includemargin),
            contact_dimension=int(contact.dim),
            condim=[int(model.geom_condim[g]) for g in geoms],
            friction=[model.geom_friction[g].tolist() for g in geoms]))
    report=dict(scope=__doc__,body_pose_approximation_permitted=a.allow_body_residual,step=step,fk_position_error_m=position_error,contacts=contacts,
                states_sha256=hashlib.sha256(raw).hexdigest())
    (a.run/"rigid-contact-audit.json").write_text(json.dumps(report,indent=2))
    for c in contacts:
        if c["distance_m"]>.0002:continue
        print(c["shapes"],c["owners"],round(c["distance_m"]*1e6,2),"um",c["condim"],
              [x.rsplit("/",1)[-1] for x in c["labels"]],c["friction"])
    sim.close()


if __name__=="__main__":main()
