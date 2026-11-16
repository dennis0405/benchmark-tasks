#!/usr/bin/env python3
"""Bounded native diagnostic for the failed three-DOF helical chain.

The exact source at SHA256 2e601bb... was executed as native trial
1d78d6a9-0ecf-4122-b160-b6d21a5113d9 and disproved the joint-ordering
hypothesis: the rotor saturated at +/-25.1327 rad while the hard-helix
residual reached 25.0026 rad.  This probe asks only whether that loss is a
three-DOF topology/constraint effect or sensitivity to the near-massless
auxiliary link.  It runs one already-supported two-DOF control and three
otherwise identical three-DOF chains with 1 mg, 10 mg, and 100 mg auxiliary
links.  Total moving mass remains 9 g in every chain.

This is discovery, not qualification.  It has no contact, fixture, preload,
friction, scorer, or task-success claim.  It never writes pose, velocity, or
joint state after reset; native velocity targets are ordinary diagnostic
actuator commands.  The frozen 0.02 rad hard-helix residual bar is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
import uuid

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--trace", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output_path = Path(args.output).resolve()
trace_path = Path(args.trace).resolve()
for path in (output_path, trace_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
trial_id = str(uuid.uuid4())
source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
app = AppLauncher(args).app


import numpy as np
import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD = 0.001
P = LEAD / (2.0 * math.pi)
STEPS_PER_PHASE = 600
AXIAL_SPEED = LEAD / 4.0
TOTAL_MOVING_MASS = 0.009
VARIANTS = {
    "control_2dof": None,
    "chain_aux_1mg": 1.0e-6,
    "chain_aux_10mg": 1.0e-5,
    "chain_aux_100mg": 1.0e-4,
}


def save(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    value = float(measured)
    passed = math.isfinite(value) and {
        "==": value == threshold,
        "<=": value <= threshold,
        ">=": value >= threshold,
    }[operator]
    return {
        "name": name,
        "measured": value,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def body(
    stage,
    root: str,
    name: str,
    mass: float,
    inertia_xyz: tuple[float, float, float],
):
    shape = UsdGeom.Cube.Define(stage, f"{root}/{name}")
    shape.CreateSizeAttr(0.01)
    shape.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(shape.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia_xyz))
    PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim()).CreateDisableGravityAttr(True)
    return shape.GetPath()


def create_variant(stage, name: str, auxiliary_mass: float | None) -> None:
    root_path = f"/World/{name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    physx_root = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    physx_root.CreateSolverPositionIterationCountAttr(128)
    physx_root.CreateSolverVelocityIterationCountAttr(4)
    frame = body(stage, root_path, "frame", 1.0, (1.0e-3, 1.0e-3, 1.0e-3))
    carrier = body(stage, root_path, "carrier", 1.0e-6, (1.0e-12, 1.0e-12, 1.0e-12))
    mount = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))
    axial = UsdPhysics.PrismaticJoint.Define(stage, f"{root_path}/axial")
    axial.CreateBody0Rel().SetTargets([frame])
    axial.CreateBody1Rel().SetTargets([carrier])
    axial.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-0.005)
    axial.CreateUpperLimitAttr(0.005)

    if auxiliary_mass is None:
        rotor_mass = TOTAL_MOVING_MASS - 1.0e-6
        rotor = body(stage, root_path, "rotor", rotor_mass, (5.0e-7, 5.0e-7, 7.0e-8))
        rotation_parent = carrier
    else:
        support = body(stage, root_path, "support", auxiliary_mass, (1.0e-12, 1.0e-12, 1.0e-12))
        rotor_mass = TOTAL_MOVING_MASS - 1.0e-6 - auxiliary_mass
        rotor = body(stage, root_path, "rotor", rotor_mass, (5.0e-7, 5.0e-7, 7.0e-8))
        rotation_parent = carrier

    rotation = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/rotation")
    rotation.CreateBody0Rel().SetTargets([rotation_parent])
    rotation.CreateBody1Rel().SetTargets([rotor if auxiliary_mass is None else support])
    rotation.CreateAxisAttr("Z")
    rotation.CreateLowerLimitAttr(-1440.0)
    rotation.CreateUpperLimitAttr(1440.0)
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(rotation.GetPrim(), "rotZ")
    mimic.CreateReferenceJointRel().SetTargets([axial.GetPath()])
    mimic.CreateReferenceJointAxisAttr("rotX")
    mimic.CreateGearingAttr(-360.0 / LEAD)
    mimic.CreateOffsetAttr(0.0)
    mimic.CreateNaturalFrequencyAttr(0.0)
    mimic.CreateDampingRatioAttr(0.0)

    axial_drive = UsdPhysics.DriveAPI.Apply(axial.GetPrim(), "linear")
    axial_drive.CreateTypeAttr("force")
    axial_drive.CreateStiffnessAttr(0.0)
    axial_drive.CreateDampingAttr(100.0)
    axial_drive.CreateMaxForceAttr(10.0)
    axial_drive.CreateTargetVelocityAttr(0.0)

    if auxiliary_mass is not None:
        elastic = UsdPhysics.PrismaticJoint.Define(stage, f"{root_path}/elastic")
        elastic.CreateBody0Rel().SetTargets([support])
        elastic.CreateBody1Rel().SetTargets([rotor])
        elastic.CreateAxisAttr("Z")
        elastic.CreateLowerLimitAttr(-0.001)
        elastic.CreateUpperLimitAttr(0.001)
        elastic_drive = UsdPhysics.DriveAPI.Apply(elastic.GetPrim(), "linear")
        elastic_drive.CreateTypeAttr("force")
        elastic_drive.CreateStiffnessAttr(1.0e6)
        elastic_drive.CreateDampingAttr(4000.0)
        elastic_drive.CreateMaxForceAttr(1000.0)
        elastic_drive.CreateTargetPositionAttr(0.0)
        elastic_drive.CreateTargetVelocityAttr(0.0)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    for name, auxiliary_mass in VARIANTS.items():
        create_variant(sim.stage, name, auxiliary_mass)
    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view unavailable")
    physics.set_subspace_roots("/")

    views = {}
    indices = wp.array([0], dtype=wp.uint32, device=args.device)
    for name, auxiliary_mass in VARIANTS.items():
        view = physics.create_articulation_view(f"/World/{name}")
        expected_dofs = 2 if auxiliary_mass is None else 3
        if view.count != 1 or view.max_dofs != expected_dofs:
            raise RuntimeError(
                f"{name}: unexpected articulation shape {view.count}x{view.max_dofs}"
            )
        names = list(view.get_metatype(0).dof_names)
        link_names = list(view.get_metatype(0).link_names)
        native_masses = wp.to_torch(view.get_masses()).detach().cpu().numpy()[0].copy()
        moving_mass = float(
            sum(native_masses[index] for index, link_name in enumerate(link_names) if link_name != "frame")
        )
        views[name] = {
            "view": view,
            "dof_names": names,
            "link_names": link_names,
            "native_moving_mass_kg": moving_mass,
            "axial": names.index("axial"),
            "rotation": names.index("rotation"),
            "elastic": names.index("elastic") if "elastic" in names else None,
            "samples": [],
        }

    for phase, velocity in (("forward", AXIAL_SPEED), ("reverse", -AXIAL_SPEED)):
        for record in views.values():
            view = record["view"]
            targets = wp.to_torch(view.get_dof_velocity_targets()).clone()
            targets[0, record["axial"]] = velocity
            view.set_dof_velocity_targets(wp.from_torch(targets.contiguous()), indices)
        for step in range(STEPS_PER_PHASE):
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            for name, record in views.items():
                view = record["view"]
                q = wp.to_torch(view.get_dof_positions()).detach().cpu().numpy()[0].copy()
                qd = wp.to_torch(view.get_dof_velocities()).detach().cpu().numpy()[0].copy()
                record["samples"].append(
                    {"phase": phase, "step": step, "q": q.tolist(), "qd": qd.tolist()}
                )

    results = {}
    checks = [
        check("actual_physics_dt_absolute_error", abs(sim.get_physics_dt() - DT), "<=", 1.0e-9, "s"),
        check("variant_count", len(views), "==", 4, "articulations"),
    ]
    for name, record in views.items():
        data = np.asarray([sample["q"] for sample in record["samples"]], dtype=np.float64)
        qd = np.asarray([sample["qd"] for sample in record["samples"]], dtype=np.float64)
        axial = data[:, record["axial"]]
        rotation = data[:, record["rotation"]]
        residual = (rotation - rotation[0]) - (axial - axial[0]) / P
        forward_travel = axial[STEPS_PER_PHASE - 1] - axial[0]
        reverse_travel = axial[-1] - axial[STEPS_PER_PHASE - 1]
        result = {
            "dof_names": record["dof_names"],
            "link_names": record["link_names"],
            "auxiliary_mass_kg": VARIANTS[name],
            "total_authored_moving_mass_kg": TOTAL_MOVING_MASS,
            "native_moving_mass_kg": record["native_moving_mass_kg"],
            "finite_state_fraction": float(np.isfinite(np.concatenate((data.ravel(), qd.ravel()))).mean()),
            "helical_residual_max_rad": float(np.max(np.abs(residual))),
            "forward_axial_travel_m": float(forward_travel),
            "reverse_axial_travel_m": float(reverse_travel),
            "rotation_range_rad": float(np.ptp(rotation)),
            "elastic_displacement_max_m": (
                float(np.max(np.abs(data[:, record["elastic"]])))
                if record["elastic"] is not None
                else 0.0
            ),
        }
        results[name] = result
        checks.extend(
            [
                check(f"{name}_finite_state_fraction", result["finite_state_fraction"], "==", 1.0, "fraction"),
                check(f"{name}_native_dof_count", len(record["dof_names"]), "==", 2 if VARIANTS[name] is None else 3, "dofs"),
                check(f"{name}_native_moving_mass_error", abs(record["native_moving_mass_kg"] - TOTAL_MOVING_MASS), "<=", 1.0e-7, "kg"),
                check(f"{name}_forward_axial_travel_min", forward_travel, ">=", 2.0e-4, "m"),
                check(f"{name}_reverse_axial_travel_max", reverse_travel, "<=", -2.0e-4, "m"),
            ]
        )
        # The frozen residual bar is a validity check for the already-supported
        # two-DOF control.  Three-DOF residuals are diagnostic outcomes, not a
        # qualification claim or a weaker alternate acceptance threshold.
        if name == "control_2dof":
            checks.append(
                check(
                    "control_2dof_helical_residual_max",
                    result["helical_residual_max_rad"],
                    "<=",
                    0.02,
                    "rad",
                )
            )

    chain_residuals = np.asarray(
        [results[name]["helical_residual_max_rad"] for name in VARIANTS if name != "control_2dof"],
        dtype=np.float64,
    )
    conditioning_span = float(np.max(chain_residuals) - np.min(chain_residuals))
    control_residual = results["control_2dof"]["helical_residual_max_rad"]
    trace_payload = {
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "qualification": False,
        "scope": "two-DOF control versus three-DOF auxiliary-mass sweep; constraint/conditioning discovery only",
        "frozen_residual_threshold_rad": 0.02,
        "results": results,
        "chain_residual_span_across_100x_aux_mass_rad": conditioning_span,
        "chain_to_control_residual_ratios": {
            name: results[name]["helical_residual_max_rad"] / max(control_residual, 1.0e-12)
            for name in VARIANTS if name != "control_2dof"
        },
        "samples": {name: record["samples"] for name, record in views.items()},
    }
    save(trace_path, trace_payload)
    artifact = str(trace_path.relative_to(output_path.parent.parent))
    save(
        output_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "qualification": False,
            "diagnostics": {
                "frozen_three_dof_residual_threshold_rad": 0.02,
                "variant_results": results,
                "chain_residual_span_across_100x_aux_mass_rad": conditioning_span,
            },
            "checks": checks,
            "overall_pass": all(item["passed"] for item in checks),
            "artifacts": [artifact],
        },
    )
    if not all(item["passed"] for item in checks):
        raise RuntimeError(
            "mimic chain conditioning probe failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, flush=True)
        if not output_path.exists():
            save(
                output_path,
                {
                    "native_trial_id": trial_id,
                    "source_sha256": source_sha256,
                    "qualification": False,
                    "error": detail,
                    "checks": [
                        {
                            "name": "native_probe_completed",
                            "measured": 0.0,
                            "operator": "==",
                            "threshold": 1.0,
                            "unit": "boolean",
                            "passed": False,
                        }
                    ],
                    "overall_pass": False,
                    "artifacts": [],
                },
            )
        raise
    finally:
        app.close()
