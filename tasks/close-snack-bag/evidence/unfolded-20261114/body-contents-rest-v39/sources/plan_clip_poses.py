"""Evaluate tool-pose reachability without changing the simulated state."""
import numpy as np
from scipy.spatial.transform import Rotation


def plan_clip_poses(c,grasp):
    ik=c.sim.arm("right")
    current=c.sim.state.joint_q.numpy().copy()
    candidates=[(x,-.02,z,0.) for x in (.54,.60,.66) for z in (.85,.90,.95)]
    candidates += [(.54,-.02,.85,yaw) for yaw in (-.18,.18)]
    output=[]
    for x,y,z,yaw in candidates:
        root=np.array([x,y,z])
        rotation=Rotation.from_euler("z",yaw)
        quat=(rotation*grasp["relative_rotation"]).as_quat()
        point=root+rotation.apply(grasp["offset"])
        target=c.takes._fold_targets(c.film,{"right":point},{"right":quat})["right"]
        with ik.host_fk():
            q=ik.solve_q_full(current,target,quat)
        actual_pos,actual_quat=ik.fk_ee(q)
        miss=float(np.linalg.norm(actual_pos-target))
        turn=float((Rotation.from_quat(actual_quat).inv()*Rotation.from_quat(quat)).magnitude())
        output.append(dict(root=root.tolist(),yaw_rad=yaw,position_error_m=miss,angle_error_rad=turn,
            feasible=miss<.004 and turn<.05,joints=q[ik.arm_coords].tolist(),
            max_joint_change_rad=float(abs(q[ik.arm_coords]-current[ik.arm_coords]).max())))
    return output
