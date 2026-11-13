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
    quat=c.takes._frame_quat((0,0,-1),(0,1,0))
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.022,0,0])
    yield from c.move({arm:handle+[0,0,.15]},{arm:quat},.08,100)
    # Read the pose again: a free upright clip may settle or tilt slightly.
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.022,0,0])
    yield from c.move({arm:handle},{arm:quat},.025,100)
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
