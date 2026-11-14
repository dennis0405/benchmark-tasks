"""Mouth lift only. Robot requests change; film is never attached."""
import numpy as np
from scipy.spatial.transform import Rotation
from pinch_lift_metrics import mouth_capture,hand_valid,MIN_AREA_MM2,confirmed_release_evidence,primitive_surface_distance


def observe(c):
    entity=c.sim.scene.by_name["pouch"];q=c.sim.state.particle_q.numpy()[entity.particles]
    owners=c.sim.model.shape_body.numpy();forces=c.shape_forces()
    reference=getattr(c,"pinch_lift_reference",None);hands={}
    bodies=c.sim.state.body_q.numpy();kinds=c.sim.model.shape_type.numpy()
    scales=c.sim.model.shape_scale.numpy();transforms=c.sim.model.shape_transform.numpy()
    for arm,panel in c.session.arms.items():
        centres,_,_=c.takes._fold_pads(c.film,arm)
        origin=centres.mean(0);rotation=Rotation.from_quat(panel.ik.ee_pose(c.sim.state)[1])
        local=rotation.inv().apply(q-origin);pads=rotation.inv().apply(centres-origin)
        low,high=sorted(pads[:,1]);capture=mouth_capture(local,entity.faces,entity.grid,[low+.002,high-.002])
        loads=[float(np.linalg.norm(forces[np.isin(owners,side)].sum(0)))
               for side in c.sim.scene.embodiment.arms[arm].finger_bodies]
        hand=dict(wall_area_mm2=capture["area_mm2"],opposed_load_N=loads,
            captured_vertices=capture["vertices"],pad_centre_m=origin.tolist(),inner_gap_m=float(high-low-.004))
        finger_bodies=np.concatenate([np.asarray(side).reshape(-1) for side in c.sim.scene.embodiment.arms[arm].finger_bodies])
        distances=np.full(len(q),np.inf)
        for sid in np.flatnonzero(np.isin(owners,finger_bodies)&np.isin(kinds,[3,4,7])):
            body=bodies[owners[sid]];br=Rotation.from_quat(body[3:])
            centre=body[:3]+br.apply(transforms[sid,:3])
            sr=br*Rotation.from_quat(transforms[sid,3:])
            distances=np.minimum(distances,primitive_surface_distance(sr.inv().apply(q-centre),int(kinds[sid]),scales[sid]))
        hand.update(finger_film_distance_m=float(distances.min()),
            wall_finger_distance_m=[float(distances[np.unique(entity.grid[layer,-5:,:])].min()) for layer in range(2)])
        if reference:
            base=reference[arm];slips=[];lifts=[];retained=[];hold_slips=[]
            hold=getattr(c,"pinch_lift_hold_local",{}).get(arm)
            for wall,(ids,points,initial_z) in enumerate(zip(base["vertices"],base["local"],base["world_z"])):
                slips.append(float(np.quantile(np.linalg.norm(local[ids]-points,axis=1),.9)))
                lifts.append(float(np.median(q[ids,2]-initial_z)))
                central=np.argsort(np.linalg.norm(points-np.median(points,axis=0),axis=1))[:5]
                landmarks=local[np.asarray(ids)[central]]
                retained.append(bool(((abs(landmarks[:,0])<=.006)&(abs(landmarks[:,2])<=.04)&
                    (landmarks[:,1]>=low+.002)&(landmarks[:,1]<=high-.002)).all()))
                if hold is not None:hold_slips.append(float(np.quantile(np.linalg.norm(local[ids]-hold[wall],axis=1),.9)))
            hand.update(wall_slip_p90_m=slips,wall_lift_m=lifts,
                wall_landmarks_retained=retained,wall_hold_slip_p90_m=hold_slips,
                hand_lift_m=float(origin[2]-base["origin"][2]))
        hands[arm]=hand
    top=float(entity.frame[2]+c.sim.snack_board_height)
    return dict(reference_locked=reference is not None,hands=hands,
                minimum_bag_height_above_board_m=float(q[:,2].min()-top))


def drive(c,hands):
    c.phase="pinch_lift_acquire"
    measurement=observe(c);entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles];reference={}
    for arm,panel in c.session.arms.items():
        hand=measurement["hands"][arm];ids=hand["captured_vertices"]
        if min(hand["wall_area_mm2"])<MIN_AREA_MM2 or min(map(len,ids))<3:
            c.failure=f"{arm} has no verified two-wall mouth grasp: {hand}";return
        origin=np.asarray(hand["pad_centre_m"]);rotation=Rotation.from_quat(panel.ik.ee_pose(c.sim.state)[1])
        local=rotation.inv().apply(q-origin)
        reference[arm]=dict(vertices=ids,local=[local[i].copy() for i in ids],
                            world_z=[q[i,2].copy() for i in ids],origin=origin)
    c.pinch_lift_reference=reference
    c.marks["pinch_lift_reference"]={a:dict(vertices=b["vertices"],origin=b["origin"].tolist(),
        local=[p.tolist() for p in b["local"]],world_z=[p.tolist() for p in b["world_z"]]) for a,b in reference.items()}
    release_streak={arm:0 for arm in hands}
    def monitor_release():
        measured=observe(c)
        for arm in hands:
            release_streak[arm]=release_streak[arm]+1 if confirmed_release_evidence(measured["hands"][arm]) else 0
            if release_streak[arm]>=max(1,round(.25*c.sim.request.physics.control_hz)):
                c.marks["pinch_lift_failure_measurement"]=measured
                c.failure=f"{arm} physically separated from the film for at least 0.25 s"
                return False
        return True
    if not all(hand_valid(observe(c)["hands"][a],lifted=False) for a in hands):
        c.failure="initial opposed film contact is not verified";return
    c.phase="pinch_lift_raise"
    advance=float(getattr(c,"lift_advance_m",0.))
    turn=float(getattr(c,"lift_pitch_deg",0.))
    points={a:np.asarray(reference[a]["origin"])+[advance,0,.05] for a in hands}
    c.marks["pinch_lift_target_delta_m"]=[advance,0,.05]
    c.marks["pinch_lift_pitch_deg"]=turn
    target_quats={a:c.takes._fold_hand(a,turn) for a in hands}
    motion=c.move(points,target_quats,.025,80,keep_target=True)
    while True:
        try:next(motion)
        except StopIteration as end:
            c.marks["pinch_lift_target_reached"]=bool(end.value)
            break
        yield
        if not monitor_release():return
    c.pinch_lift_hold_local={}
    q=c.sim.state.particle_q.numpy()[entity.particles]
    for arm,base in reference.items():
        origin=c.takes._fold_jaws(c.film,arm)
        rotation=Rotation.from_quat(c.session.arms[arm].ik.ee_pose(c.sim.state)[1])
        local=rotation.inv().apply(q-origin)
        c.pinch_lift_hold_local[arm]=[local[ids].copy() for ids in base["vertices"]]
    c.marks["pinch_lift_hold_reference_local"]={a:[p.tolist() for p in walls] for a,walls in c.pinch_lift_hold_local.items()}
    c.phase="pinch_lift_hold"
    for _ in c.wait(210):
        yield
        if not monitor_release():return
    c.phase="pinch_lift_done"
