"""CPU-authored cutaway fixture proposal, NOT a validated Isaac environment.

The form follows a two-jaw angular gripper with a central piston and adjustable
stop. Dimensions/internal gearing are benchmark assumptions, not DRG 80 CAD.
Runtime helical coupling/friction, valves, scoring and robot control are absent.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from geometry_seed import asset_usd, box, cylinder, ring_sectors


def shift_mesh(piece, x, y):
    piece = copy.deepcopy(piece)
    piece["points"] = [(p[0] + x, p[1] + y, p[2]) for p in piece["points"]]
    piece["footprint"] = [(p[0] + x, p[1] + y) for p in piece["footprint"]]
    return piece


def fixture_parts():
    bore_y = -0.018

    def cover_outer(theta):
        co, si = math.cos(theta), math.sin(theta)
        rx = 0.05 / abs(co) if abs(co) > 1e-12 else math.inf
        ry = (
            (0.0358 - bore_y if si > 0 else 0.0358 + bore_y) / abs(si)
            if abs(si) > 1e-12
            else math.inf
        )
        return min(rx, ry)

    cover = [
        shift_mesh(p, 0, bore_y)
        for p in ring_sectors("Cover", cover_outer, lambda _: 0.0044, 0.080, 0.090)
    ]
    barrel = [
        shift_mesh(p, 0, bore_y)
        for p in ring_sectors("Barrel", lambda _: 0.016, lambda _: 0.009, 0.010, 0.080)
    ]
    frame = [box("Base", (0.100, 0.0716, 0.010), (0, 0, 0.005))]
    frame += [
        box("LeftSupport", (0.016, 0.0716, 0.070), (-0.042, 0, 0.045)),
        box("RightSupport", (0.016, 0.0716, 0.070), (0.042, 0, 0.045)),
    ]
    frame += barrel + cover
    parts = {
        "frame": {
            "pieces": frame,
            "mass_kg": 1.0,
            "com": (0, 0, 0.045),
            "inertia": (0.002, 0.002, 0.002),
            "datums": {"bore_axis": (0, bore_y, 0), "cover_top_z": 0.09},
        }
    }
    for name, sign in (("left_jaw", 1), ("right_jaw", -1)):
        pieces = [
            box("PivotBlock", (0.018, 0.024, 0.012), (0, 0, 0)),
            box("Shoulder", (0.022, 0.012, 0.012), (sign * 0.011, 0, 0.006)),
            box("Finger", (0.008, 0.012, 0.050), (sign * 0.015, 0, 0.025)),
            box("Pad", (0.012, 0.012, 0.012), (sign * 0.023, 0, 0.054)),
        ]
        parts[name] = {
            "pieces": pieces,
            "mass_kg": 0.06,
            "com": (sign * 0.012, 0, 0.025),
            "inertia": (0.000040, 0.000040, 0.000012),
            "datums": {"hinge": (0, 0, 0)},
        }
    parts["piston"] = {
        "pieces": [cylinder("Piston", 0.008, 0.020, (0, 0, -0.010))],
        "mass_kg": 0.03,
        "com": (0, 0, -0.010),
        "inertia": (0.0000015, 0.0000015, 0.0000010),
        "datums": {"stop_contact": (0, 0, 0)},
    }
    return parts


def fixture_layout():
    return {
        "frame": ("parts/frame.usda", (0, 0, 0)),
        "left_jaw": ("parts/left_jaw.usda", (-0.030, 0.022, 0.095)),
        "right_jaw": ("parts/right_jaw.usda", (0.030, 0.022, 0.095)),
        "piston": ("parts/piston.usda", (0, -0.018, 0.072)),
        "adjustment_screw": ("../assets/socket_adjuster.usda", (0, -0.018, 0.1017)),
        "nut73": ("../assets/thin_nut.usda", (0, -0.018, 0.0930)),
        "nut74": ("../assets/thin_nut.usda", (0, -0.018, 0.0997)),
    }


def body(stage, name, reference, pos, kinematic=False):
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = stage.DefinePrim("/Bench/" + name, "Xform")
    prim.GetReferences().AddReference(reference)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if kinematic:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    return prim


def connect(joint, body0, body1, anchor0, anchor1=(0, 0, 0)):
    from pxr import Gf, Sdf

    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Bench/" + body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Bench/" + body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*anchor1))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(True)


def drive(joint, axis, stiffness, damping, cap, target=0):
    from pxr import UsdPhysics

    api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
    api.CreateTypeAttr("force")
    api.CreateStiffnessAttr(stiffness)
    api.CreateDampingAttr(damping)
    api.CreateMaxForceAttr(cap)
    api.CreateTargetPositionAttr(target)
    api.CreateTargetVelocityAttr(0)


def cylindrical(stage, name, parent, child, anchor0):
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint.Define(stage, "/Bench/" + name)
    connect(joint, parent, child, anchor0)
    for axis in ("transX", "transY", "rotX", "rotY"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1)
        limit.CreateHighAttr(-1)
    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "transZ")
    limit.CreateLowAttr(-0.005)
    limit.CreateHighAttr(0.005)
    drive(joint, "transZ", 20000, 20, 25)
    # Runtime must update helix targets and dissipative friction references from
    # measured state, not the task goal. These are inert draft defaults only.
    drive(joint, "rotZ", 1, 0.001, 0.001 if name == "screw_guide" else 0.004)
    return joint


def physx_coupling(stage, name, kind, a, b, relationships, ratio_name, ratio):
    from pxr import Gf, Sdf

    prim = stage.DefinePrim("/Bench/" + name, kind)
    for field, target in {"body0": a, "body1": b, **relationships}.items():
        prim.CreateRelationship("physics:" + field).SetTargets([Sdf.Path("/Bench/" + target)])
    prim.CreateAttribute("physics:" + ratio_name, Sdf.ValueTypeNames.Float).Set(ratio)
    prim.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)
    # Proposed local-X coupling frames: align to the actual Y hinge/Z slide.
    # PhysX parsing and the ratio sign remain native-probe requirements.
    q_y_axis = Gf.Quatf(math.sqrt(0.5), Gf.Vec3f(0, 0, math.sqrt(0.5)))
    q_z_axis = Gf.Quatf(math.sqrt(0.5), Gf.Vec3f(0, -math.sqrt(0.5), 0))
    prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(q_y_axis)
    prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(
        q_y_axis if kind == "PhysxPhysicsGearJoint" else q_z_axis
    )
    return prim


def validate_static_graph(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 7
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    anchors = []
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            valid, reason = mesh.ValidateTopology(
                mesh.GetFaceVertexIndicesAttr().Get(),
                mesh.GetFaceVertexCountsAttr().Get(),
                len(mesh.GetPointsAttr().Get()),
            )
            assert valid, reason
            mesh_count += 1
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        assert len(a) == len(b) == 1
        pa, pb = stage.GetPrimAtPath(a[0]), stage.GetPrimAtPath(b[0])
        assert pa.HasAPI(UsdPhysics.RigidBodyAPI) and pb.HasAPI(UsdPhysics.RigidBodyAPI)
        va = cache.GetLocalToWorldTransform(pa).Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        vb = cache.GetLocalToWorldTransform(pb).Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        error = (va - vb).GetLength()
        assert error < 1e-7, (prim.GetPath(), error)
        anchors.append({"joint": str(prim.GetPath()), "initial_anchor_error_m": error})
    assert len(anchors) == 6
    return {
        "status": "CPU composition, mesh topology and joint-anchor checks ONLY; NOT native physics",
        "openusd_version": list(Usd.GetVersion()),
        "rigid_bodies": len(bodies),
        "meshes_checked": mesh_count,
        "anchors": anchors,
        "physx_couplings": [
            {"path": str(p.GetPath()), "proposed_type": p.GetTypeName()}
            for p in stage.Traverse()
            if p.GetTypeName() in {"PhysxPhysicsGearJoint", "PhysxPhysicsRackAndPinionJoint"}
        ],
    }


def export(root):
    from pxr import Usd, UsdPhysics

    root = Path(root).resolve()
    part_dir = root / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = fixture_parts()
    for name, asset in parts.items():
        (part_dir / (name + ".usda")).write_text(asset_usd(asset))
    stage = Usd.Stage.CreateNew(str(root / "bench.usda"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    top = stage.DefinePrim("/Bench", "Xform")
    stage.SetDefaultPrim(top)
    layout = fixture_layout()
    for name, (reference, pos) in layout.items():
        body(stage, name, reference, pos, kinematic=name == "frame")
    cylindrical(stage, "screw_guide", "frame", "adjustment_screw", layout["adjustment_screw"][1])
    nut_anchor = tuple(n - s for n, s in zip(layout["nut73"][1], layout["adjustment_screw"][1]))
    cylindrical(stage, "nut_guide", "adjustment_screw", "nut73", nut_anchor)
    retained = UsdPhysics.FixedJoint.Define(stage, "/Bench/nut74_retained")
    connect(retained, "adjustment_screw", "nut74", (0, 0, -0.0020))
    for name, low, high in (("left_jaw", -90, 0), ("right_jaw", 0, 90)):
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Bench/" + name + "_hinge")
        connect(joint, "frame", name, layout[name][1])
        joint.CreateAxisAttr("Y")
        joint.CreateLowerLimitAttr(low)
        joint.CreateUpperLimitAttr(high)
    slider = UsdPhysics.PrismaticJoint.Define(stage, "/Bench/piston_slide")
    connect(slider, "frame", "piston", layout["piston"][1])
    slider.CreateAxisAttr("Z")
    slider.CreateLowerLimitAttr(0)
    slider.CreateUpperLimitAttr(0.010)
    drive(slider, "linear", 0, 5, 0)  # Valve/runtime intentionally absent.
    physx_coupling(
        stage,
        "jaw_pair",
        "PhysxPhysicsGearJoint",
        "left_jaw",
        "right_jaw",
        {"hinge0": "left_jaw_hinge", "hinge1": "right_jaw_hinge"},
        "gearRatio",
        1,
    )
    physx_coupling(
        stage,
        "piston_to_jaw",
        "PhysxPhysicsRackAndPinionJoint",
        "left_jaw",
        "piston",
        {"hinge": "left_jaw_hinge", "prismatic": "piston_slide"},
        "ratio",
        -150,
    )
    stage.GetRootLayer().Save()
    scene_file = root / "bench.usda"
    scene_file.write_text(scene_file.read_text().rstrip() + "\n")
    report = validate_static_graph(stage)
    (root / "cpu-fixture-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    source = Path(__file__)
    manifest = {
        "status": "CPU geometry/joint-graph proposal; NOT native-validated or a runnable task",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "forms": "Representative cutaway angular-gripper bench, not exact DRG 80 geometry",
        "units": "m, kg; angular drive/limit targets follow USD degrees",
        "assumptions": "All dimensions, masses, springs, friction caps, internal transmission and nut74 placement are benchmark proposals. Rack ratio sign, coupling, contact, material properties and unit conversions need native verification.",
        "layout": {
            name: {"reference": ref, "position": pos} for name, (ref, pos) in layout.items()
        },
        "parts": parts,
        "missing_runtime": [
            "two helical spring constraints and reaction torques",
            "dissipative thread-friction updates",
            "pressure/vent control geometry and dynamics",
            "scorer/reset/measurement integration",
            "tool racks and YAM control",
        ],
    }
    (root / "fixture-proposal.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return stage, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "fixture")
    stage, _ = export(parser.parse_args().output)
    print(stage.GetRootLayer().realPath)
