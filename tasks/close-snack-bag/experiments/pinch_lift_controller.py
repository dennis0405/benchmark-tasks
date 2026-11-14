"""Vertical mouth lift only. Robot requests change; film is never attached."""
import numpy as np
from scipy.spatial.transform import Rotation
from pinch_lift_metrics import mouth_capture,hand_valid,MIN_AREA_MM2


def observe(c):
    entity=c.sim.scene.by_name["pouch"];q=c.sim.state.particle_q.numpy()[entity.particles]
    owners=c.sim.model.shape_body.numpy();forces=c.shape_forces()
    reference=getattr(c,"pinch_lift_reference",None);hands={}
    for arm,panel in c.session.arms.items():
        centres,_,_=c.takes._fold_pads(c.film,arm)
        origin=centres.mean(0);rotation=Rotation.from_quat(panel.ik.ee_pose(c.sim.state)[1])
        local=rotation.inv().apply(q-origin);pads=rotation.inv().apply(centres-origin)
        low,high=sorted(pads[:,1]);capture=mouth_capture(local,entity.faces,entity.grid,[low+.002,high-.002])
        loads=[float(np.linalg.norm(forces[np.isin(owners,side)].sum(0)))
               for side in c.sim.scene.embodiment.arms[arm].finger_bodies]
        hand=dict(wall_area_mm2=capture["area_mm2"],opposed_load_N=loads,
            captured_vertices=capture["vertices"],pad_centre_m=origin.tolist(),inner_gap_m=float(high-low-.004))
        if reference:
            base=reference[arm];slips=[];lifts=[]
            for ids,points,initial_z in zip(base["vertices"],base["local"],base["world_z"]):
                slips.append(float(np.quantile(np.linalg.norm(local[ids]-points,axis=1),.9)))
                lifts.append(float(np.median(q[ids,2]-initial_z)))
            hand.update(wall_slip_p90_m=slips,wall_lift_m=lifts,
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
    def retained(lifted=False):
        measured=observe(c)
        good=all(hand_valid(measured["hands"][a],lifted=lifted) for a in hands)
        if not good:c.marks["pinch_lift_failure_measurement"]=measured
        return good
    if not retained():c.failure="initial opposed film contact is not verified";return
    c.phase="pinch_lift_raise"
    points={a:np.asarray(reference[a]["origin"])+[0,0,.05] for a in hands}
    motion=c.move(points,hands,.025,80,keep_target=True)
    while True:
        try:next(motion)
        except StopIteration as end:
            if not end.value:c.failure="vertical lift target was not reached";return
            break
        yield
        if not retained():c.failure="two-wall grasp lost during vertical lift";return
    c.phase="pinch_lift_hold"
    for _ in c.wait(210):
        yield
        if not retained(lifted=True):c.failure="lifted material did not remain held by both hands";return
    c.phase="pinch_lift_done"
