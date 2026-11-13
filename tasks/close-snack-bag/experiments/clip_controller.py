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
    panel.grip_request=.4
    # +X approach with pads parting along Y is reachable at this bench.
    # A downward tool axis saturated the wrist limits without approaching
    # the handle, so verify the measured pad location before closing.
    quat=c.takes._frame_quat((1,0,0),(0,1,0))
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.050,0,0])
    yield from c.move({arm:handle+[0,0,.15]},{arm:quat},.08,100)
    # Read the pose again: a free upright clip may settle or tilt slightly.
    pose=clip_pose(c)
    handle=pose[:3]+Rotation.from_quat(pose[3:]).apply([-.050,0,0])
    yield from c.move({arm:handle},{arm:quat},.025,100)
    miss=float(np.linalg.norm(c.takes._fold_jaws(c.film,arm)-handle))
    print("CLIP_REACH_ERROR_M",miss,flush=True)
    if miss > .006:
        c.failure=f"clip handle reach error {miss:.4f} m exceeds 6 mm"
        return None
    c.phase="clip_squeeze_handles"
    hz=float(c.sim.request.physics.control_hz)
    request=.4
    for _ in range(c.frames(240)):
        if clip_angle(c)<=-.34:
            # Hold the measured slide position instead of driving through
            # the clip's hinge limit. The 0.1 mm relief avoids extra squeeze.
            coords=c.sim.scene.embodiment.arms[arm].gripper_coords
            actual=float(c.sim.state.joint_q.numpy()[coords][0])
            panel.grip_request=float(np.clip(1-(actual+.0001)/.0475,0,1))
            break
        request=min(.85,request+.15/hz)
        panel.grip_request=request
        yield
    yield from c.wait(30)
    angle=clip_angle(c)
    print("CLIP_OPEN",angle,clip_pose(c).tolist(),flush=True)
    if not -.445 <= angle <= -.15:
        c.failure=f"handle squeeze did not retain a valid opening: {angle:.3f} rad"
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
    return grasp


def clip_grasp_drive(controller):
    yield from controller.retract_grasp("right")
    grasp=yield from grasp_clip(controller)
    if grasp is not None:
        controller.phase="clip_carry_rotate"
        root=np.array([.55,-.005,.90])
        reached=yield from move_clip_root(controller,root,Rotation.identity(),grasp,.05)
        if not reached:
            return
        measured=clip_pose(controller)
        miss=float(np.linalg.norm(measured[:3]-root))
        turn=float(Rotation.from_quat(measured[3:]).magnitude())
        print("CLIP_CARRY",dict(position_error_m=miss,angle_error_rad=turn),flush=True)
        if miss>.006 or turn>.10:
            controller.failure="clip carry did not reach the requested pose"
            return
        controller.phase="clip_grasp_watch"
        yield from controller.wait(180)


def move_clip_root(c,target,rotation,grasp,speed=.025):
    """Measured-pose corrections act on the robot cursor, never the clip."""
    quat=(rotation*grasp["relative_rotation"]).as_quat()
    yield from c.move({"right":target+rotation.apply(grasp["offset"])},{"right":quat},speed,150)
    for _ in range(4):
        pose=clip_pose(c)
        actual_rotation=Rotation.from_quat(pose[3:])
        error=np.asarray(target)-pose[:3]
        angle=float((rotation.inv()*actual_rotation).magnitude())
        if np.linalg.norm(error)<.002 and angle<.08:
            break
        jaw=c.takes._fold_jaws(c.film,"right")
        ee_rotation=Rotation.from_quat(c.session.arms["right"].ik.ee_pose(c.sim.state)[1])
        relative=actual_rotation.inv()*ee_rotation
        quat=(rotation*relative).as_quat()
        yield from c.move({"right":jaw+np.clip(error,-.008,.008)},{"right":quat},min(speed,.015),100)
    pose=clip_pose(c)
    miss=float(np.linalg.norm(np.asarray(target)-pose[:3]))
    actual_rotation=Rotation.from_quat(pose[3:])
    angle=float((rotation.inv()*actual_rotation).magnitude())
    print("CLIP_POSE_ERROR",dict(phase=c.phase,position_m=miss,angle_rad=angle),flush=True)
    if miss>.006 or angle>.15:
        c.failure=f"clip pose missed target during {c.phase}: {miss:.4f} m, {angle:.3f} rad"
        return False
    grasp["offset"]=actual_rotation.inv().apply(c.takes._fold_jaws(c.film,"right")-pose[:3])
    grasp["relative_rotation"]=actual_rotation.inv()*Rotation.from_quat(c.session.arms["right"].ik.ee_pose(c.sim.state)[1])
    return True


def regrasp_fold(c, holder, presser):
    """Pinch all folded layers at an unsupported side margin, then lift.

    This component test does not attach film nodes or restore a folded state.
    """
    arm="left"
    if arm==presser:
        yield from c.lift_press(arm)
    else:
        yield from c.retract_grasp(arm)
        yield from c.press(arm,leave_down=True)
        yield from c.lift_press(arm)
    c.phase="fold_stack_approach"
    panel=c.session.arms[arm]
    panel.grip_request=.6
    quat=c.takes._fold_hand(arm)
    top,line,yc=(c.layout[k] for k in ("board_top","line","center_y"))
    # In this reachable side-facing wrist frame the 80 mm pad axis is Y.
    # Its inner end must clear the board at Y=60 mm; the 12 mm pad width
    # along X must overlap the folded band beyond the X=570 mm crease.
    yield from c.move({arm:(line+.010,yc+.18,top+.10)},{arm:quat},.05,100)
    yield from c.move({arm:(line+.010,yc+.16,top+.010)},{arm:quat},.025,100)
    point=np.array([line+.010,yc+.105,top+.010])
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
        yield from c.press("right",leave_down=True)
        yield from c.lift_press("right")
    else:
        yield from c.lift_press("right")
    c.phase="fold_stack_lift"
    jaw=c.takes._fold_jaws(c.film,arm)
    yield from c.move({arm:jaw+[0,0,.025]},{arm:quat},.015,100)
    from fold_metrics import fold_metrics
    entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles]
    fold=fold_metrics(q,entity.grid,[(34,28,40)])["folds"][0]
    print("STACK_FOLD_AFTER_LIFT",fold,flush=True)
    if not fold["valid"] or fold["median_angle_rad"]<2.5 or fold["folded_fraction"]<.8:
        c.failure="folded stack unfolded during transfer/lift"
        return False
    if c.after_fold!="seal":
        c.phase="fold_stack_watch"
        yield from c.wait(180)
    return True


def seal_fold(c):
    """Attempt insertion using measured cloth/clip poses, then release.

    Completion of these commands is not a success score. Independent contacts,
    cloth geometry and the full hands-free interval must still be assessed.
    """
    grasp=yield from grasp_clip(c)
    if grasp is None:
        return
    entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles]
    crease=q[entity.grid[:,34,:]]
    center=np.median(crease[:,8:25],axis=(0,1))
    y_axis=np.mean(crease[:,24],axis=0)-np.mean(crease[:,8],axis=0)
    y_axis/=np.linalg.norm(y_axis)
    body=np.median(q[entity.grid[:,28,8:25]],axis=(0,1))
    x_axis=body-center
    x_axis-=y_axis*np.dot(x_axis,y_axis)
    x_axis/=np.linalg.norm(x_axis)
    z_axis=np.cross(x_axis,y_axis)
    if z_axis[2]<0:
        c.failure="fold insertion frame has a downward normal"
        return
    rotation=Rotation.from_matrix(np.column_stack((x_axis,y_axis,z_axis)))
    local=rotation.inv().apply(q-center)
    near=(abs(local[:,1])<.05)&(local[:,0]>-.003)&(local[:,0]<.012)
    if not near.any():
        c.failure="no measured folded film in the insertion band"
        return
    bottom=float(local[near,2].min())
    # The closed lower inner face is -0.2 mm in clip_spring_v2.xml.
    target=center+rotation.apply([-.035,0,bottom-.0003])
    stage=target-x_axis*.040+z_axis*.040
    c.phase="clip_stage"
    if not (yield from move_clip_root(c,stage,rotation,grasp,.045)):
        return
    stage=target-x_axis*.040
    c.phase="clip_descend"
    if not (yield from move_clip_root(c,stage,rotation,grasp,.015)):
        return
    c.phase="clip_insert"
    if not (yield from move_clip_root(c,target,rotation,grasp,.010)):
        return
    miss=float(np.linalg.norm(target-clip_pose(c)[:3]))
    print("INSERT_REACH_ERROR_M",miss,flush=True)
    if miss>.006:
        c.failure=f"clip insertion error {miss:.4f} m exceeds 6 mm"
        return
    c.phase="clip_release"
    quat=c.session.arms["right"].ik.ee_pose(c.sim.state)[1]
    c.session.arms["right"].grip_request=.7
    yield from c.wait(60)
    jaw=c.takes._fold_jaws(c.film,"right")
    yield from c.move({"right":jaw-x_axis*.080+z_axis*.020},{"right":quat},.025,80)
    c.phase="fold_stack_release"
    c.session.arms["left"].grip_request=.95
    yield from c.wait(30)
    left_quat=c.session.arms["left"].ik.ee_pose(c.sim.state)[1]
    jaw=c.takes._fold_jaws(c.film,"left")
    yield from c.move({"left":jaw+[0,.025,.10]},{"left":left_quat},.035,100)
    c.session.arms["left"].grip_request=.6
    c.phase="hands_free_hold"
    yield from c.wait(210)
