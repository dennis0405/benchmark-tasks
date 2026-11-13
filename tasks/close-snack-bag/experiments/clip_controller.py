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
    c.support_holds.pop(arm,None)
    panel=c.session.arms[arm]
    c.phase="clip_approach"
    panel.grip_request=0.
    # +X approach with pads parting along Y is reachable at this bench.
    # A downward tool axis saturated the wrist limits without approaching
    # the handle, so verify the measured pad location before closing.
    nominal=Rotation.from_quat(c.takes._frame_quat((1,0,0),(0,1,0)))
    parking=clip_pose(c)[:3].copy()
    def target():
        pose=clip_pose(c)
        rotation=Rotation.from_quat(pose[3:])
        quat=(rotation*Rotation.from_euler("x",-np.pi/2)*nominal).as_quat()
        return pose[:3]+rotation.apply([-.060,0,0]),quat
    handle,quat=target()
    yield from c.move({arm:handle+[0,0,.15]},{arm:quat},.15,100)
    if np.linalg.norm(clip_pose(c)[:3]-parking)>.05:
        c.failure="clip displaced more than 50 mm before descent"
        return None
    # Read the pose again: a free upright clip may settle or tilt slightly.
    handle,quat=target()
    yield from c.move({arm:handle+[0,0,.020]},{arm:quat},.10,80)
    if np.linalg.norm(clip_pose(c)[:3]-parking)>.05:
        c.failure="clip displaced more than 50 mm during approach"
        return None
    handle,quat=target()
    yield from c.move({arm:handle},{arm:quat},.025,100)
    for _ in range(3):
        if np.linalg.norm(clip_pose(c)[:3]-parking)>.05:
            c.failure="clip displaced more than 50 mm before grasp"
            return None
        handle,quat=target()
        if np.linalg.norm(c.takes._fold_jaws(c.film,arm)-handle)<.003:
            break
        yield from c.move({arm:handle},{arm:quat},.015,80)
    # Validate against the CURRENT object pose, not the earlier cursor goal.
    handle,quat=target()
    miss=float(np.linalg.norm(c.takes._fold_jaws(c.film,arm)-handle))
    print("CLIP_REACH_ERROR_M",miss,flush=True)
    if miss > .006:
        c.failure=f"clip handle reach error {miss:.4f} m exceeds 6 mm"
        return None
    c.phase="clip_squeeze_handles"
    hz=float(c.sim.request.physics.control_hz)
    request=0.
    closing_start=clip_pose(c)[:3].copy()
    for _ in range(c.frames(240)):
        if np.linalg.norm(clip_pose(c)[:3]-closing_start)>.05:
            c.failure="clip displaced more than 50 mm while squeezing"
            return None
        if clip_angle(c)<=-.32:
            # Preserve the spring-limited pinch; do not load the soft
            # hinge stop to make transport look stable.
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
    if not -.405 <= angle <= -.15:
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
    yield from c.move({arm:jaws+[0,0,.020]},{arm:grip_quat},.035,80)
    yield from c.move({arm:jaws+[0,0,.09]},{arm:grip_quat},.10,100)
    pose=clip_pose(c)
    print("CLIP_LIFT",pose.tolist(),"angle",clip_angle(c),flush=True)
    if pose[2]-grasp["start_z"] < .05:
        c.failure="clip did not rise at least 50 mm with the gripper"
        return None
    rotation=Rotation.from_quat(pose[3:])
    grasp["offset"]=rotation.inv().apply(c.takes._fold_jaws(c.film,arm)-pose[:3])
    grasp["relative_rotation"]=rotation.inv()*Rotation.from_quat(panel.ik.ee_pose(c.sim.state)[1])
    c.marks["clip_grasp_transform"]=dict(offset=grasp["offset"].tolist(),
        relative_quat=grasp["relative_rotation"].as_quat().tolist(),clip_pose=pose.tolist())
    return grasp


def clip_grasp_drive(controller):
    yield from controller.retract_grasp("right")
    grasp=yield from grasp_clip(controller)
    if grasp is not None:
        if getattr(controller,"plan_clip",False):
            from plan_clip_poses import plan_clip_poses
            controller.phase="clip_pose_plan"
            controller.marks["clip_pose_plan"]=plan_clip_poses(controller,grasp)
            return
        controller.phase="clip_carry_rotate"
        root=np.asarray(getattr(controller,"carry_root",(.66,-.02,.95)),dtype=float)
        via=getattr(controller,"carry_via_root",None)
        if via is not None:
            controller.phase="clip_carry_via"
            if not (yield from move_clip_root(controller,np.asarray(via),Rotation.identity(),grasp,.05)):
                return
            controller.phase="clip_carry_rotate"
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


def move_clip_root(c,target,rotation,grasp,speed=.025,precise=False):
    """Measured-pose corrections act on the robot cursor, never the clip."""
    quat=(rotation*grasp["relative_rotation"]).as_quat()
    def guarded(point,quat,speed,settle):
        for _ in c.move({"right":point},{"right":quat},speed,settle):
            yield
            pose=clip_pose(c)
            expected=pose[:3]+Rotation.from_quat(pose[3:]).apply(grasp["offset"])
            if np.linalg.norm(expected-c.takes._fold_jaws(c.film,"right"))>.030:
                c.failure=f"clip slipped more than 30 mm during {c.phase}"
                return False
        return True
    if not (yield from guarded(target+rotation.apply(grasp["offset"]),quat,speed,150)):
        return False
    for _ in range(12 if precise else 4):
        pose=clip_pose(c)
        actual_rotation=Rotation.from_quat(pose[3:])
        error=np.asarray(target)-pose[:3]
        angle=float((rotation.inv()*actual_rotation).magnitude())
        if np.linalg.norm(error)<(.0005 if precise else .002) and angle<(.01 if precise else .08):
            break
        jaw=c.takes._fold_jaws(c.film,"right")
        ee_rotation=Rotation.from_quat(c.session.arms["right"].ik.ee_pose(c.sim.state)[1])
        relative=actual_rotation.inv()*ee_rotation
        quat=(rotation*relative).as_quat()
        if not (yield from guarded(jaw+np.clip(error,-.008,.008),quat,min(speed,.015),100)):
            return False
    pose=clip_pose(c)
    miss=float(np.linalg.norm(np.asarray(target)-pose[:3]))
    actual_rotation=Rotation.from_quat(pose[3:])
    angle=float((rotation.inv()*actual_rotation).magnitude())
    print("CLIP_POSE_ERROR",dict(phase=c.phase,position_m=miss,angle_rad=angle),flush=True)
    if miss>(.001 if precise else .006) or angle>(.02 if precise else .15):
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
        yield from c.press(arm,leave_down=True,near=.010,along_fold=True)
        yield from c.lift_press(arm)
    c.phase="fold_stack_approach"
    panel=c.session.arms[arm]
    panel.grip_request=.6
    quat=c.takes._fold_hand(arm)
    top,line,yc=(c.layout[k] for k in ("board_top","line","center_y"))
    from stack_grasp import select_stack_grasp,captured_plies
    entity=c.sim.scene.by_name["pouch"]
    def measured_point():
        q=c.sim.state.particle_q.numpy()[entity.particles]
        selected=select_stack_grasp(q,entity.grid,entity.faces)
        if selected is not None:
            c.marks["stack_grasp_plan"]=selected
            print("STACK_PLAN",selected,flush=True)
            return np.asarray(selected["point"])
        c.failure="no accessible four-ply side patch in the measured folded film"
        return None
    point=measured_point()
    if point is None:return False
    # In this reachable side-facing wrist frame the 80 mm pad axis is Y.
    # Its inner end must clear the board at Y=60 mm; the 12 mm pad width
    # along X must overlap the folded band beyond the X=570 mm crease.
    yield from c.move({arm:(point[0],yc+.18,top+.10)},{arm:quat},.10,100)
    point=measured_point()
    if point is None:return False
    yield from c.move({arm:(point[0],yc+.14,point[2])},{arm:quat},.10,100)
    point=measured_point()
    if point is None:return False
    yield from c.move({arm:point},{arm:quat},.025,100)
    miss=float(np.linalg.norm(c.takes._fold_jaws(c.film,arm)-point))
    print("STACK_REACH_ERROR_M",miss,flush=True)
    if miss>.006:
        c.failure=f"fold stack reach error {miss:.4f} m exceeds 6 mm"
        return
    c.phase="fold_stack_pinch"
    for request in np.linspace(.6,1.,c.frames(45)):
        panel.grip_request=float(request)
        yield
    solver=c.sim.pipeline.program.solver
    owners=c.sim.model.shape_body.numpy()
    samples=[]
    for _ in c.wait(45):
        yield
        forces=solver.shape_contact_forces()
        samples.append([float(np.linalg.norm(forces[np.isin(owners,side)].sum(axis=0)))
                       for side in c.sim.scene.embodiment.arms[arm].finger_bodies])
    loads=np.mean(samples,axis=0).tolist()
    print("STACK_PINCH_CONTROL_SAMPLE_MEAN_N",loads,flush=True)
    captured=captured_plies(c,arm)
    print("STACK_CAPTURED_PLY_AREA_MM2",captured,flush=True)
    if min(loads)<.05:
        c.failure="folded stack has no bilateral contact across the observation window"
        return
    if min(captured)<.5:
        c.failure="gripper gap does not contain all four measured film ply groups"
        return False
    if holder=="right":
        yield from c.retract_grasp("right")
        yield from c.press("right",leave_down=True,near=.010,along_fold=True)
        if c.failure:return False
        yield from c.lift_press("right")
    else:
        yield from c.lift_press("right")
    if getattr(c,"table_insert",False):
        return True
    c.phase="fold_stack_lift"
    jaw=c.takes._fold_jaws(c.film,arm)
    yield from c.move({arm:jaw+[0,0,.040]},{arm:quat},.025,100)
    # Check actual film clearance for the 6 mm lower jaw. A one-sided lift
    # tilts the band; its lower end, not its centre, determines insertion room.
    for attempt in range(5):
        q=c.sim.state.particle_q.numpy()[entity.particles]
        band=q[entity.grid[:,32:39,:]].reshape(-1,3)
        center_y=float(np.median(q[entity.grid[:,34,:]][:,:,1]))
        selected=band[abs(band[:,1]-center_y)<.055]
        clearance=float(selected[:,2].min()-top) if len(selected) else -1.
        print("STACK_LOWEST_CLEARANCE_M",clearance,flush=True)
        if clearance>=.014:break
        if attempt==4:
            c.failure="insufficient measured clearance for the lower clip jaw"
            return False
        jaw=c.takes._fold_jaws(c.film,arm)
        yield from c.move({arm:jaw+[0,0,.020]},{arm:quat},.025,80)
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


def prepare_table_fold(c,holder,presser):
    """Press with the left pad outside the 100 mm jaw span; never lift film."""
    if presser=="left":
        # Left already presses the far side. Release the remaining raw-rim
        # grip and finish the centre with the right pad.
        yield from c.retract_grasp("right")
        yield from c.press("right",leave_down=True,near=.010,along_fold=True)
        yield from c.lift_press("right")
    else:
        # Right supports one side while the released left hand finishes
        # the centre. Then place the left pad beyond the clip's width.
        yield from c.retract_grasp("left")
        yield from c.press("left",leave_down=True,near=.010,along_fold=True)
        if c.failure:return False
        c.phase="left_side_press_transfer"
        held_config=c.support_holds.pop("left",None)
        quat=c.takes._fold_hand("left")
        jaw=c.takes._fold_jaws(c.film,"left")
        yield from c.move({"left":jaw+[0,0,.004]},{"left":quat},.04,60)
        target=np.array([c.layout["line"]+.012,c.layout["center_y"]+.100,jaw[2]+.004])
        yield from c.move({"left":target},{"left":quat},.08,80)
        yield from c.move({"left":target-[0,0,.004]},{"left":quat},.015,80)
        if held_config is not None:
            c.start_support("left",floor_z=held_config["floor_z"],quat=quat)
        yield from c.lift_press("right")
    return (yield from table_fold_gate(c))


def sequential_table_fold(c):
    """Release raw rim grips, then press the free flap down across its width.

    Pressing the crease nose while the other rim was still gripped peeled
    the flap up. A pitched tip now starts at the measured free rim instead.
    """
    yield from c.retract_grasp("left")
    yield from c.retract_grasp("right")
    right=c.session.arms["right"]
    quat=right.ik.ee_pose(c.sim.state)[1]
    jaw=c.takes._fold_jaws(c.film,"right")
    yield from c.move({"right":jaw+[0,-.10,.05]},{"right":quat},.12,100)
    yield from c.press("left",leave_down=True,near=-.045,sweep_end=.070,at_rim=True)
    if c.failure:return False
    return (yield from table_fold_gate(c))


def sequential_component(c):
    held=yield from sequential_table_fold(c)
    if held:yield from seal_fold(c)


def table_fold_gate(c):
    entity=c.sim.scene.by_name["pouch"]
    from fold_metrics import fold_metrics
    q=c.sim.state.particle_q.numpy()[entity.particles]
    fold=fold_metrics(q,entity.grid,[(34,28,40)])["folds"][0]
    print("TABLE_FOLD_BEFORE_CLIP",fold,flush=True)
    if not fold["valid"] or fold["median_angle_rad"]<2.2 or fold["folded_fraction"]<.6:
        c.failure="table-supported centre press did not retain the folded mouth"
        return False
    if getattr(c,"insert_mode","overhang")!="wedge" and c.layout.get("support_front",c.layout["line"])<c.layout["line"]-.005:
        if not (yield from drag_fold_to_edge(c)):
            return False
    c.marks["left_press_hold"]=True
    c.phase="table_fold_ready"
    yield from c.wait(3)
    return True


def drag_fold_to_edge(c):
    """Slide the supported folded neck to the edge using left-pad friction."""
    entity=c.sim.scene.by_name["pouch"]
    front=c.layout["support_front"]
    quat=np.asarray(c.support_holds["left"]["quat"]) if "left" in c.support_holds else c.session.arms["left"].ik.ee_pose(c.sim.state)[1]
    initial=None
    c.phase="fold_drag_to_edge"
    for _ in range(35):
        q=c.sim.state.particle_q.numpy()[entity.particles]
        crease=float(np.median(q[entity.grid[:,34,8:25],0]))
        if initial is None:initial=crease
        if crease<=front-.0045:
            c.marks["fold_drag_m"]=initial-crease
            return True
        jaw=c.takes._fold_jaws(c.film,"left")
        yield from c.move({"left":jaw-[.001,0,0]},{"left":quat},.015,30)
    c.failure="left-pad friction did not bring the measured fold to the board edge"
    return False


def centre_press_component(c):
    """Continue from a recorded state after the right raw-rim release."""
    yield from c.press("right",leave_down=True,near=.010,along_fold=True)
    if c.failure:return
    yield from c.lift_press("right")
    held=yield from table_fold_gate(c)
    if held:yield from seal_fold(c)


def raw_release_component(c):
    """Compare release styles from the same physically recorded rim grip."""
    yield from c.retract_grasp("right")
    yield from centre_press_component(c)


def edge_drag_component(c):
    """Establish a measured press, then test frictional transfer and clipping."""
    c.phase="left_support_contact"
    if c.support_hold_enabled:
        established=0
        for _ in range(c.frames(600)):
            load=c.support_holds.get("left",{}).get("filtered",0.)
            established=established+1 if load>=.2 else 0
            if established>=c.frames(30):break
            yield
        else:
            c.failure="left support did not establish a measured film contact"
            return
    else:
        yield from c.wait(90)
    held=yield from table_fold_gate(c)
    if held:yield from seal_fold(c)


def press_drag_component(c):
    """Re-seat the left support with its actual pitched pressing geometry."""
    right=c.session.arms["right"]
    quat=right.ik.ee_pose(c.sim.state)[1]
    jaw=c.takes._fold_jaws(c.film,"right")
    yield from c.move({"right":jaw+[0,-.12,.04]},{"right":quat},.12,100)
    yield from c.press("left",leave_down=True,near=.055)
    if c.failure:return
    yield from edge_drag_component(c)


def blade_offsets(c,rotation):
    """Actual lower-blade box corners, excluding the handles outside the board."""
    m=c.sim.model
    scale,local=m.shape_scale.numpy(),m.shape_transform.numpy()
    corners=[]
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for i,label in enumerate(m.shape_label):
        if str(label).endswith("/lower_jaw") or "/lower_tip_" in str(label):
            points=Rotation.from_quat(local[i,3:]).apply(signs*scale[i])+local[i,:3]
            corners.extend(rotation.apply(points))
    if not corners:raise ValueError("clip has no measured lower blade boxes")
    return np.asarray(corners)


def insert_over_support(c,grasp,center):
    """Enter below an overhanging fold, lift clear, then advance over the board."""
    rotation=Rotation.identity();top=c.layout["board_top"];front=c.layout.get("support_front",c.layout["line"])
    final_x=float(center[0]-.035)
    lift_x=min(final_x-.008,front-.045)
    outside_x=lift_x-.025
    # First finish rotation in the validated high workspace. Rotating
    # while approaching the lower stage let the handle slip in rigid-only
    # tests; the high-then-low route retained the spring-limited grip.
    c.phase="clip_carry_high"
    if not (yield from move_clip_root(c,np.array([.66,center[1],top+.072]),rotation,grasp,.05)):
        return None
    c.phase="clip_stage"
    if not (yield from move_clip_root(c,np.array([outside_x,center[1],top+.04]),rotation,grasp,.045)):
        return None
    if getattr(c,"insert_mode","overhang")=="wedge":
        # The 0.2 mm stepped toe approaches above the board, then lifts the
        # bottom film on contact. No frictional pull on the free flap and no
        # prescribed film motion. All insertion geometry stays observed.
        c.phase="clip_descend_outside"
        pose=clip_pose(c)
        safe_z=float(top+.0004-blade_offsets(c,Rotation.from_quat(pose[3:]))[:,2].min())
        for _ in range(4):
            if not (yield from move_clip_root(c,np.array([outside_x,center[1],safe_z]),rotation,grasp,.010,precise=True)):
                return None
            pose=clip_pose(c)
            clearance=float(pose[2]+blade_offsets(c,Rotation.from_quat(pose[3:]))[:,2].min()-top)
            if clearance>=.0002:break
            safe_z+=.0004-clearance
        else:
            c.failure="wedge lower blade did not clear the board";return None
        c.phase="clip_insert"
        target=np.array([final_x,center[1],safe_z])
        if not (yield from move_clip_root(c,target,rotation,grasp,.003,precise=True)):
            return None
        return target
    entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles]
    near=(abs(q[:,1]-center[1])<.052)&(q[:,0]>center[0]-.006)&(q[:,0]<center[0]+.012)
    if not near.any():
        c.failure="no measured film at the support insertion edge";return None
    actual_rotation=Rotation.from_quat(clip_pose(c)[3:])
    low_z=float(q[near,2].min()-.0005-blade_offsets(c,actual_rotation)[:,2].max())
    c.phase="clip_descend_outside"
    if not (yield from move_clip_root(c,np.array([outside_x,center[1],low_z]),rotation,grasp,.015,precise=True)):
        return None
    c.phase="clip_enter_under_fold"
    if not (yield from move_clip_root(c,np.array([lift_x,center[1],low_z]),rotation,grasp,.008,precise=True)):
        return None
    pose=clip_pose(c)
    safe_z=max(float(pose[2]),float(top+.0004-blade_offsets(c,Rotation.from_quat(pose[3:]))[:,2].min()))
    c.phase="clip_raise_over_edge"
    for _ in range(4):
        if not (yield from move_clip_root(c,np.array([lift_x,center[1],safe_z]),rotation,grasp,.008,precise=True)):
            return None
        pose=clip_pose(c)
        clearance=float(pose[2]+blade_offsets(c,Rotation.from_quat(pose[3:]))[:,2].min()-top)
        print("BLADE_LOWEST_CLEARANCE_M",clearance,flush=True)
        if clearance>=.0002:break
        safe_z+=.0004-clearance
    else:
        c.failure="lower blade did not clear the support before advancing";return None
    c.phase="clip_insert"
    target=np.array([final_x,center[1],safe_z])
    if not (yield from move_clip_root(c,target,rotation,grasp,.008,precise=True)):
        return None
    return target


def insert_component(c):
    """Resume the measured robot/clip grasp from a recorded component state."""
    pose=clip_pose(c);rotation=Rotation.from_quat(pose[3:])
    quat=c.session.arms["right"].ik.ee_pose(c.sim.state)[1]
    grasp=dict(offset=rotation.inv().apply(c.takes._fold_jaws(c.film,"right")-pose[:3]),
               relative_rotation=rotation.inv()*Rotation.from_quat(quat),start_z=float(pose[2]))
    yield from seal_fold(c,grasp=grasp)


def clear_left_for_clip(c):
    """Park the creasing arm outside the right arm's transport corridor."""
    yield from c.lift_press("left")
    c.phase="clear_left_for_clip"
    ik=c.sim.arm("left");initial=c.sim.state.joint_q.numpy().copy()
    current_quat=c.session.arms["left"].ik.ee_pose(c.sim.state)[1]
    chosen=None
    # Retain the current wrist first: the fixed inward pinch quaternion
    # saturated the wrist at the former side-park goal. Plan by FK/IK,
    # without writing the candidate joint vector into the live state.
    for quat in (current_quat,c.takes._fold_hand("left")):
        for x,y,z in ((.55,.22,.13),(.50,.25,.15),(.60,.18,.13)):
            target=np.array([x,c.layout["center_y"]+y,c.layout["board_top"]+z])
            tcp=c.takes._fold_targets(c.film,{"left":target},{"left":quat})["left"]
            with ik.host_fk():candidate=ik.solve_q_full(initial,tcp,quat)
            actual,turn=ik.fk_ee(candidate)
            miss=float(np.linalg.norm(actual-tcp))
            angle=float((Rotation.from_quat(turn).inv()*Rotation.from_quat(quat)).magnitude())
            if miss<.003 and angle<.05:
                chosen=(target,quat)
                c.marks["left_clear_kinematic_plan"]=dict(jaw_m=target.tolist(),tcp_residual_m=miss,
                    angle_residual_rad=angle,candidate_joints=candidate[ik.arm_coords].tolist())
                break
        if chosen is not None:break
    if chosen is None:
        c.failure="no reachable side park for the left arm";return False
    target,quat=chosen
    if not (yield from c.move({"left":target},{"left":quat},.10,150)):
        c.failure="left arm did not reach its clear side park";return False
    c.marks["left_cleared_for_clip"]=True
    c.marks["left_press_hold"]=False
    from fold_metrics import fold_metrics
    entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles]
    fold=fold_metrics(q,entity.grid,[(34,28,40)])["folds"][0]
    if fold["median_angle_rad"]<2.2 or fold["folded_fraction"]<.6:
        c.failure="fold did not retain the pre-clip gate after the pressing arm cleared"
        return False
    return True


def seal_fold(c,grasp=None):
    """Attempt insertion using measured cloth/clip poses, then release.

    Completion of these commands is not a success score. Independent contacts,
    cloth geometry and the full hands-free interval must still be assessed.
    """
    if getattr(c,"table_insert",False) and not c.marks.get("left_cleared_for_clip"):
        if not (yield from clear_left_for_clip(c)):return
    if grasp is None:grasp=yield from grasp_clip(c)
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
        # The material grid runs toward world -Y in this reset; its index
        # order must not turn an otherwise valid insertion frame upside down.
        y_axis=-y_axis
        z_axis=-z_axis
    if z_axis[2]<.2 and not getattr(c,"table_insert",False):
        c.failure="fold insertion frame is too steep for the clip path"
        return
    rotation=Rotation.from_matrix(np.column_stack((x_axis,y_axis,z_axis)))
    local=rotation.inv().apply(q-center)
    near=(abs(local[:,1])<.05)&(local[:,0]>-.003)&(local[:,0]<.012)
    if not near.any():
        c.failure="no measured folded film in the insertion band"
        return
    bottom=float(local[near,2].min())
    # The closed lower inner face is -0.2 mm in clip_spring_v2.xml.
    target=center+rotation.apply([-.035,0,bottom-.003])
    if getattr(c,"table_insert",False):
        rotation=Rotation.identity()
        x_axis,y_axis,z_axis=np.eye(3)
        target=yield from insert_over_support(c,grasp,center)
        if target is None:return
    else:
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
    yield from release_clip(c,rotation)


def release_clip(c,rotation):
    """Clear both closed handle ends before retreating the open gripper."""
    x_axis=rotation.apply([1,0,0]);z_axis=rotation.apply([0,0,1])
    c.phase="clip_release"
    quat=c.session.arms["right"].ik.ee_pose(c.sim.state)[1]
    # The opened handles' midpoint is below the root. A 95 mm gap only
    # clears the 86 mm closed ends when centred about the closed clip,
    # not about that earlier squeeze midpoint.
    c.session.arms["right"].grip_request=0.
    yield from c.wait(30)
    pose=clip_pose(c);root_rotation=Rotation.from_quat(pose[3:])
    jaw=c.takes._fold_jaws(c.film,"right")
    local=root_rotation.inv().apply(jaw-pose[:3]);local[2]=0.
    c.phase="clip_centre_open_gripper"
    if not (yield from c.move({"right":pose[:3]+root_rotation.apply(local)},{"right":quat},.008,100)):
        c.failure="open gripper could not centre clear of both clip handle ends";return
    yield from c.wait(30)
    jaw=c.takes._fold_jaws(c.film,"right")
    c.phase="clip_retreat"
    yield from c.move({"right":jaw-x_axis*.080},{"right":quat},.025,80)
    jaw=c.takes._fold_jaws(c.film,"right")
    yield from c.move({"right":jaw+z_axis*.040},{"right":quat},.05,80)
    c.phase="fold_stack_release"
    if c.marks.get("left_cleared_for_clip"):
        pass
    elif c.marks.get("left_press_hold"):
        yield from c.lift_press("left")
    else:
        c.session.arms["left"].grip_request=.95
        yield from c.wait(30)
        left_quat=c.session.arms["left"].ik.ee_pose(c.sim.state)[1]
        jaw=c.takes._fold_jaws(c.film,"left")
        yield from c.move({"left":jaw+[0,.025,.10]},{"left":left_quat},.035,100)
    c.session.arms["left"].grip_request=.6
    c.phase="hands_free_hold"
    yield from c.wait(210)
