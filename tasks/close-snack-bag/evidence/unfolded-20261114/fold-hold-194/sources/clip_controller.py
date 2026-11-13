"""Robot-only commands for grasping the clip by its handle ends."""
import numpy as np
from scipy.spatial.transform import Rotation


def clip_pose(controller):
    entity=controller.sim.scene.by_name["clip"]
    body=controller.sim.state.body_q.numpy()
    return body[entity.bodies[0]].copy()


def clip_angle(controller):
    entity=controller.sim.scene.by_name["clip"]
    body=controller.sim.state.body_q.numpy()
    relative=Rotation.from_quat(body[entity.bodies[0],3:]).inv()*Rotation.from_quat(body[entity.bodies[1],3:])
    return float(relative.as_rotvec()[1])


def grasp_clip(controller, arm="right"):
    """Return a measured grasp transform through the generator's return value."""
    c=controller
    panel=c.session.arms[arm]
    c.phase="clip_approach"
    panel.grip_request=.6
    # +X approach with pads parting along Y is reachable at this bench.
    # A downward tool axis saturated the wrist limits without approaching
    # the handle, so verify the measured pad location before closing.
    quat=c.takes._frame_quat((1,0,0),(0,1,0))
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.022,0,0])
    yield from c.move({arm:handle+[0,0,.15]},{arm:quat},.08,100)
    # Read the pose again: a free upright clip may settle or tilt slightly.
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.022,0,0])
    yield from c.move({arm:handle},{arm:quat},.025,100)
    miss=float(np.linalg.norm(c.takes._fold_jaws(c.film,arm)-handle))
    print("CLIP_REACH_ERROR_M",miss,flush=True)
    if miss > .006:
        c.failure=f"clip handle reach error {miss:.4f} m exceeds 6 mm"
        return None
    c.phase="clip_squeeze_handles"
    panel.grip_request=1.
    yield from c.wait(90)
    angle=clip_angle(c)
    print("CLIP_OPEN",angle,clip_pose(c).tolist(),flush=True)
    if angle > -.06:
        c.failure="handle squeeze did not open the clip"
        return None
    pose=clip_pose(c)
    rotation=Rotation.from_quat(pose[3:])
    jaws=c.takes._fold_jaws(c.film,arm)
    grip_quat=panel.ik.ee_pose(c.sim.state)[1]
    grasp=dict(offset=rotation.inv().apply(jaws-pose[:3]),
               relative_rotation=rotation.inv()*Rotation.from_quat(grip_quat),
               start_z=float(pose[2]))
    c.phase="clip_lift"
    yield from c.move({arm:jaws+[0,0,.09]},{arm:grip_quat},.035,100)
    pose=clip_pose(c)
    print("CLIP_LIFT",pose.tolist(),"angle",clip_angle(c),flush=True)
    if pose[2]-grasp["start_z"] < .05:
        c.failure="clip did not rise at least 50 mm with the gripper"
        return None
    c.phase="clip_rotate"
    desired=grasp["relative_rotation"].as_quat()
    yield from c.move({arm:c.takes._fold_jaws(c.film,arm)},{arm:desired},.04,120)
    return grasp


def clip_grasp_drive(controller):
    yield from controller.retract_grasp("right")
    grasp=yield from grasp_clip(controller)
    if grasp is not None:
        controller.phase="clip_grasp_watch"
        yield from controller.wait(180)


def regrasp_fold(c, holder, presser):
    """Pinch all folded layers at an unsupported side margin, then lift.

    This component test does not attach film nodes or restore a folded state.
    """
    arm="left"
    if arm==presser:
        yield from c.lift_press(arm)
    else:
        yield from c.retract_grasp(arm)
    c.phase="fold_stack_approach"
    panel=c.session.arms[arm]
    panel.grip_request=.6
    quat=c.takes._frame_quat((1,0,0),(0,0,-1))
    top,line,yc=(c.layout[k] for k in ("board_top","line","center_y"))
    yield from c.move({arm:(line-.015,yc+.18,top+.10)},{arm:quat},.05,100)
    yield from c.move({arm:(line-.015,yc+.12,top+.010)},{arm:quat},.025,100)
    point=np.array([line-.015,yc+.075,top+.010])
    yield from c.move({arm:point},{arm:quat},.015,100)
    miss=float(np.linalg.norm(c.takes._fold_jaws(c.film,arm)-point))
    print("STACK_REACH_ERROR_M",miss,flush=True)
    if miss>.006:
        c.failure=f"fold stack reach error {miss:.4f} m exceeds 6 mm"
        return
    c.phase="fold_stack_pinch"
    panel.grip_request=1.
    yield from c.wait(90)
    solver=c.sim.pipeline.program.solver
    forces=solver.shape_contact_forces()
    owners=c.sim.model.shape_body.numpy()
    loads=[float(np.linalg.norm(forces[np.isin(owners,side)].sum(axis=0)))
           for side in c.sim.scene.embodiment.arms[arm].finger_bodies]
    print("STACK_PINCH_LOADS_N",loads,flush=True)
    if min(loads)<.05:
        c.failure="folded stack has no measured bilateral finger contact"
        return
    if holder=="right":
        yield from c.retract_grasp("right")
    else:
        yield from c.lift_press("right")
    c.phase="fold_stack_lift"
    jaw=c.takes._fold_jaws(c.film,arm)
    yield from c.move({arm:jaw+[0,0,.025]},{arm:quat},.015,100)
    c.phase="fold_stack_watch"
    yield from c.wait(180)
