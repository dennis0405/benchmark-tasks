"""Diagnostic: initialize an open clip astride a pre-folded mouth and release it.

This deliberately places bodies by state assignment to inspect contact geometry. It is
not a robot action or a demonstration and cannot establish end-to-end feasibility.
"""

from pathlib import Path

import mujoco

def jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def probe(held_open):
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("prototype.xml")))
    data = mujoco.MjData(model)
    fold = model.jnt_qposadr[jid(model, "fold")]
    fold2 = model.jnt_qposadr[jid(model, "fold2")]
    hinge = model.jnt_qposadr[jid(model, "clip_hinge")]
    clip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clip")
    clip_free = model.jnt_qposadr[model.body_jntadr[clip_body]]
    data.qpos[fold] = -3.14
    data.qpos[fold2] = 3.14
    data.qpos[clip_free:clip_free + 7] = [0.22, 0.09, 0.0455, 0, 0, 0, 1]
    data.qpos[hinge] = -0.35
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        if held_open:
            data.ctrl[1] = -0.025
        mujoco.mj_step(model, data)
    pairs = set()
    for contact in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (contact.geom1, contact.geom2)]
        pairs.add(tuple(sorted(names)))
    bag_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bag")
    initial = data.xpos[clip_body, 0] - data.xpos[bag_body, 0]
    # Pull the clip away from the fold, identically for the two conditions.
    for _ in range(500):
        data.xfrc_applied[clip_body, 0] = 0.05
        if held_open:
            data.ctrl[1] = -0.025
        mujoco.mj_step(model, data)
    shift = data.xpos[clip_body, 0] - data.xpos[bag_body, 0] - initial
    return float(data.qpos[hinge]), float(shift), sorted(pair for pair in pairs if any("jaw" in name for name in pair))


def run():
    shifts = {}
    for label, held_open in (("spring released", False), ("hinge held open", True)):
        angle, shift, pairs = probe(held_open)
        shifts[label] = abs(shift)
        print(f"{label}: hinge={angle:.4f} rad, relative displacement={shift * 1000:.2f} mm")
        print("  pre-pull jaw contacts:", pairs)
    if shifts["spring released"] >= shifts["hinge held open"]:
        print("INCONCLUSIVE: the released spring did not resist this pull better than the open control.")
    print("Diagnostic only: no robot, no validated real-world retention threshold or benchmark score.")


if __name__ == "__main__":
    run()
