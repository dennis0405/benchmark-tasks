"""Primitive support geometry in a proposed YAM TCP orientation."""
import numpy as np
import newton
from scipy.spatial.transform import Rotation


def low_offset(sim,arm,quat,ik):
    """Lowest primitive support point in a proposed TCP frame, including XY."""
    m=sim.model;b=sim.state.body_q.numpy()
    tcp,actual=ik.ee_pose(sim.state)
    delta=Rotation.from_quat(quat)*Rotation.from_quat(actual).inv()
    fingers=[int(x) for side in sim.scene.embodiment.arms[arm].finger_bodies for x in np.asarray(side).reshape(-1)]
    owners=m.shape_body.numpy();kinds=m.shape_type.numpy();scales=m.shape_scale.numpy();local=m.shape_transform.numpy()
    points=[]
    for sid,owner in enumerate(owners):
        if int(owner) not in fingers:continue
        kind=int(kinds[sid]);sc=scales[sid]
        tf=local[sid];rot=Rotation.from_quat(b[owner,3:]);center=b[owner,:3]+rot.apply(tf[:3])
        rotation=rot*Rotation.from_quat(tf[3:])
        if kind==int(newton.GeoType.BOX):
            corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*sc
            points.extend(delta.apply(center+rotation.apply(corners)-tcp))
        elif kind in (int(newton.GeoType.CAPSULE),int(newton.GeoType.SPHERE)):
            h=float(sc[1]) if kind==int(newton.GeoType.CAPSULE) else 0.
            ends=center+rotation.apply([[0,0,-h],[0,0,h]])
            points.extend(delta.apply(ends-tcp)-[0,0,float(sc[0])])
    points=np.asarray(points)
    return points[points[:,2].argmin()]


def plan_press_pose(sim,arm,ik,takes,film,surface_point,floor_z,advance,travel):
    """Choose a reachable approach, contact and stroke end without live FK writes."""
    tcp,actual_quat=ik.ee_pose(sim.state)
    jaws=takes._fold_jaws(film,arm)
    local_jaw=Rotation.from_quat(actual_quat).inv().apply(jaws-tcp)
    initial=sim.state.joint_q.numpy().copy();sign=1. if arm=="left" else -1.
    attempts=[]
    settings=[(0.,25.,sign),(0.,15.,sign),(-30.*sign,25.,sign),(30.*sign,25.,sign),
              (-60.*sign,25.,sign),(60.*sign,25.,sign),(0.,25.,-sign),(0.,15.,-sign)]
    for yaw,pitch,closing in settings:
        y,p=np.deg2rad([yaw,pitch])
        quat=takes._frame_quat((np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),-np.sin(p)),
                               (-closing*np.sin(y),closing*np.cos(y),0.))
        rotation=Rotation.from_quat(quat)
        jaw_offset=rotation.apply(local_jaw)
        lowest=low_offset(sim,arm,quat,ik)-jaw_offset
        point=np.asarray(surface_point)-lowest
        goals=[point+[0,0,.0465],point,
               point+[advance,travel,floor_z-surface_point[2]]]
        q=initial.copy();errors=[]
        for goal in goals:
            target=goal-jaw_offset
            with ik.host_fk():q=ik.solve_q_full(q,target,quat)
            actual,turn=ik.fk_ee(q)
            errors.append(dict(position_m=float(np.linalg.norm(actual-target)),
                angle_rad=float((Rotation.from_quat(turn).inv()*rotation).magnitude())))
        attempt=dict(yaw_deg=yaw,pitch_deg=pitch,closing_sign=closing,errors=errors)
        attempts.append(attempt)
        if all(e["position_m"]<.002 and e["angle_rad"]<.03 for e in errors):
            return dict(quat=quat,point=point,lowest=lowest,attempts=attempts,selected=attempt)
    return dict(attempts=attempts,selected=None)

