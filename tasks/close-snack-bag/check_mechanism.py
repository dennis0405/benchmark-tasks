"""Check settling, flap rotation, and the clip's spring return (no robot or scoring)."""

from pathlib import Path
import math

import mujoco
import numpy as np


def run():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("prototype.xml")))
    data = mujoco.MjData(model)
    clip_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "clip_hinge")
    clip_index = model.jnt_qposadr[clip_joint]
    fold_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fold")
    fold_index = model.jnt_qposadr[fold_joint]
    for _ in range(500):
        mujoco.mj_step(model, data)
    settled = data.qpos.copy()
    assert np.isfinite(settled).all(), "non-finite position during settling"
    for _ in range(500):
        data.ctrl[1] = -0.07
        mujoco.mj_step(model, data)
    opened = float(data.qpos[clip_index])
    for _ in range(1000):
        data.ctrl[1] = 0.0
        mujoco.mj_step(model, data)
    returned = float(data.qpos[clip_index])
    tip_lift_mm = 44.0 * math.sin(-opened)
    assert tip_lift_mm >= 20.0, f"jaw tip did not lift 20 mm: {tip_lift_mm:.2f} mm"
    assert returned > opened + 0.1, f"clip did not spring back: {returned:.3f} rad"
    print(f"Mechanism check passed: opened {opened:.3f} rad ({tip_lift_mm:.2f} mm tip lift), released {returned:.3f} rad")
    for _ in range(2000):
        data.ctrl[0] = -0.025
        mujoco.mj_step(model, data)
    folded = float(data.qpos[fold_index])
    assert folded < -2.8, f"flap did not fold across the mouth: {folded:.3f} rad"
    print(f"Flap joint reached {folded:.3f} rad under direct test torque")
    print("This is an actuator-driven mechanism check, not a robot demonstration.")


if __name__ == "__main__":
    run()
