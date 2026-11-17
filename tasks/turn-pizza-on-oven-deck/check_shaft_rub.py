"""Native isolated control: drive the inverted shaft across the top of a free pizza."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from grading import Grade
from isaac_geometry import build_food, build_peel, build_station, xf, box, mat
from layout import MOUTH, OVEN_Y, STONE_Z


def main():
    dt = 1 / 240
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=dt))
    stage = omni.usd.get_context().get_stage()
    build_station(stage)
    bodies = build_food(stage)
    build_peel(stage)
    center = np.array([MOUTH + 0.23, OVEN_Y])
    # The inverted blade clears the food vertically. The shaft spans the
    # diameter. A diagnostic fixture drives the peel, never the food.
    angle = 0.0
    height = STONE_Z + 0.076

    peel_path = "/World/Peel"
    prim = stage.GetPrimAtPath(peel_path)
    initial = np.r_[
        center + [0.215, 0], height, Rotation.from_euler("x", math.pi).as_quat()
    ]
    UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(*initial[:3]))
    UsdGeom.Xformable(prim).AddOrientOp().Set(
        Gf.Quatf(initial[6], Gf.Vec3f(*initial[3:6]))
    )
    # In this contact-control fixture a revolute joint supplies pure yaw;
    # the prismatic guide leaves vertical motion free, so the original 400 g
    # tool supplies its own normal load under gravity.
    for wall in ("back", "left", "right", "roof"):
        stage.RemovePrim("/World/Oven/" + wall)
    pivot = Gf.Vec3f(*center, height)
    carrier = "/World/ShaftControl"
    xf(stage, carrier, tuple(pivot), 0.2)
    spin = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/driveShaft")
    spin.CreateBody1Rel().SetTargets([carrier])
    spin.CreateLocalPos0Attr(pivot)
    spin.CreateAxisAttr("Z")
    drive = UsdPhysics.DriveAPI.Apply(spin.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateDampingAttr(0.02)
    drive.CreateStiffnessAttr(1.0)
    drive.CreateTargetPositionAttr(0.0)
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/shaftFloat")
    slide.CreateBody0Rel().SetTargets([carrier])
    slide.CreateBody1Rel().SetTargets([peel_path])
    slide.CreateLocalPos0Attr(Gf.Vec3f(0.215, 0, 0))
    slide.CreateLocalRot1Attr(Gf.Quatf(0, Gf.Vec3f(1, 0, 0)))
    slide.CreateAxisAttr("Z")
    slide.CreateLowerLimitAttr(-0.024)
    slide.CreateUpperLimitAttr(0.010)
    # A real contact filter far from the fixture keeps the direct-food-contact
    # test live. Grasp history is supplied explicitly at the start of this control.
    xf(stage, "/World/IdleProbe", (0, 0, 2), 1.0, True)
    box(
        stage,
        "/World/IdleProbe/shape",
        (0, 0, 0),
        (0.01, 0.01, 0.01),
        (0.5, 0.5, 0.5),
        mat(stage, "Probe", 0.4),
    )
    sim.reset()
    pv = sim.physics_manager.get_physics_sim_view()
    grade = Grade(pv, bodies, ["/World/IdleProbe"])
    for _ in range(240):
        sim.step(render=False)
    grade.reset()
    grade.had_grasp = True
    peak_load = 0.0
    contacted_steps = 0
    legacy_contact_angle = 0.0
    load = 0.0
    for step in range(2700):
        if 480 <= step < 2400:
            angle += dt * 15
        drive.GetTargetPositionAttr().Set(angle)
        sim.step(render=False)
        prev = grade.angle
        grade.observe()
        forces = (
            grade.contacts.get_contact_force_matrix(dt=dt)
            .numpy()
            .reshape(-1, grade.nfilters, 3)
        )
        load = float(np.abs(forces[:, 1, 2]).sum())
        peak_load = max(peak_load, load)
        if np.linalg.norm(forces[:, 1], axis=1).sum() > 0.015:
            contacted_steps += 1
            legacy_contact_angle += abs(grade.angle - prev)
    result = grade.result()
    result.update(
        fixture="revolute-driven, vertically free inverted shaft, dynamic pizza; grasp-history precondition true",
        shaft_load_peak_N=peak_load,
        contacted_steps=contacted_steps,
        old_contact_angle_deg=legacy_contact_angle,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
    assert abs(grade.angle) >= 45 and legacy_contact_angle >= 45 and contacted_steps > 0
    assert grade.had_grasp and grade.fault is None
    assert grade.score == 0 and grade.engaged_steps == 0 and grade.top_steps > 0


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    app.close(exit_code=1)
    raise
else:
    app.close()
