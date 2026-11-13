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
    points=[]
    for sid,owner in enumerate(m.shape_body.numpy()):
        if int(owner) not in fingers:continue
        kind=int(m.shape_type.numpy()[sid]);sc=m.shape_scale.numpy()[sid]
        tf=m.shape_transform.numpy()[sid];rot=Rotation.from_quat(b[owner,3:]);center=b[owner,:3]+rot.apply(tf[:3])
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

