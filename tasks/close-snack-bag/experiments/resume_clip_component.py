"""Reset a clip-only diagnostic from an observed fold checkpoint.

Particle/joint velocities and plastic rest angles are preserved. Solver
contact/friction caches are rebuilt, so this is a component reset, never a
claim of an uninterrupted end-to-end demonstration.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import newton


def restore(sim,session,directory):
    directory=Path(directory)
    raw=(directory/"clip-entry.npz").read_bytes()
    metadata=json.loads((directory/"clip-entry.json").read_text())
    import io
    data=np.load(io.BytesIO(raw))
    entity=sim.scene.by_name["pouch"]
    for key in ("q","b","j","d","v","r"):
        if key not in data or not np.isfinite(data[key]).all():
            raise ValueError(f"missing/invalid physical checkpoint array: {key}")
    if not np.array_equal(data["faces"],entity.faces) or not np.array_equal(data["grid"],entity.grid):
        raise ValueError("checkpoint topology differs from the rebuilt scene")
    if len(data["j"])!=len(sim.state.joint_q) or data["b"].shape!=sim.state.body_q.numpy().shape:
        raise ValueError("checkpoint rigid model differs from the rebuilt scene")
    for state in (sim.pipeline.state_0,sim.pipeline.state_1):
        q=state.particle_q.numpy();q[entity.particles]=data["q"];state.particle_q.assign(q)
        v=state.particle_qd.numpy();v[entity.particles]=data["v"];state.particle_qd.assign(v)
        state.particle_f.zero_();state.body_f.zero_()
        state.joint_q.assign(data["j"]);state.joint_qd.assign(data["d"])
        newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
    pos_error=float(abs(sim.state.body_q.numpy()[:,:3]-data["b"][:,:3]).max())
    quaternion_dot=abs((sim.state.body_q.numpy()[:,3:]*data["b"][:,3:]).sum(1))
    if pos_error>1e-5 or np.any(quaternion_dot<1-1e-5):
        raise ValueError("checkpoint FK is inconsistent with the rebuilt model")
    rest=entity._rest.numpy();rest[entity.rows]=data["r"];entity._rest.assign(rest)
    if "targets" in data:
        targets=data["targets"].copy()
        target_source="checkpoint array"
    else:
        # The first checkpoint recorder saved physical states before it
        # saved control targets. Reconstruct actuated entries from its
        # contemporaneous trace, never from measured joint positions.
        trace=json.loads((directory/"trace.json").read_text())
        row=next(t for t in trace if t["step"]==metadata["step"])
        targets=sim.control.joint_target_q.numpy().copy()
        for arm in ("left","right"):
            targets[sim.arm(arm).arm_coords]=row["robot_targets"][arm]["joints"]
            targets[sim.scene.embodiment.arms[arm].gripper_coords]=row["robot_targets"][arm]["gripper_joint_targets"]
        target_source="actuated entries from the source checkpoint frame trace"
    if not np.isfinite(targets).all():raise ValueError("invalid checkpoint controls")
    sim.control.joint_target_q.assign(targets)
    sim.pipeline.reset_solvers(flags=0);sim.pipeline.reset_finite_flags()
    sim.pipeline.invalidate_graph()
    for arm,panel in session.arms.items():
        panel.on_reset({}, {})
        ik=sim.arm(arm)
        goal=data["j"].copy();goal[ik.arm_coords]=targets[ik.arm_coords]
        pos,quat=ik.fk_ee(goal)
        panel.aim(pos,quat,moving=False)
        coord=sim.scene.embodiment.arms[arm].gripper_coords[0]
        panel.grip_request=float(np.clip(1-targets[coord]/.0475,0,1))
    return dict(layout=metadata["layout"],source_step=metadata["step"],
                left_cleared_for_clip=metadata.get("left_cleared_for_clip",False),
                support_holds=metadata.get("support_holds",{}),
                seed_sha256=hashlib.sha256(raw).hexdigest(),fk_position_error_m=pos_error,
                target_source=target_source,target_q=targets.tolist(),
                limitation=__doc__.strip())
