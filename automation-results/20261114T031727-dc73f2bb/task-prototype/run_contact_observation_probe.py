#!/usr/bin/env python3
"""Trusted-host native GPU probe for PhysX contact observation.

This deliberately small scene proves the two contact paths used by the task:
the legacy report-event subscription and the pinned PhysX tensor contact view.
It never substitutes configured forces for observations.  The tensor rows retain
the sensor/filter actor ordering, contact point, normal, separation, force and
the exact integration interval used to convert force to impulse.
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "measurements/contact-observation-probe.json")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    app = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=args.headless).app
        import omni.usd
        import warp as wp
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager
        from omni.physx import get_physx_simulation_interface

        dt = 1.0 / 240.0
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)

        floor = UsdGeom.Cube.Define(stage, "/World/Probe/target")
        floor.CreateSizeAttr(1.0)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.74))
        floor.AddScaleOp().Set(Gf.Vec3d(0.16, 0.16, 0.02))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

        ball = UsdGeom.Sphere.Define(stage, "/World/Probe/sensor")
        ball.CreateRadiusAttr(0.015)
        ball.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.84))
        UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
        UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(0.10)
        PhysxSchema.PhysxContactReportAPI.Apply(ball.GetPrim()).CreateThresholdAttr(0.0)

        callback_errors = []
        callback_rows = []
        step = [1]
        def post_cb(value):
            try:
                callback_rows.append({"step": step[0], "dt_float32_s": float(value)})
                step[0] += 1
            except Exception as exc:
                callback_errors.append(f"{type(exc).__name__}: {exc}")
        post_handle = PhysxManager.register_callback(
            post_cb, IsaacEvents.POST_PHYSICS_STEP,
            name="helicoil_contact_probe_post", wrap_weak_ref=False,
        )

        event_rows = []
        event_errors = []
        def sdf(value):
            return str(PhysicsSchemaTools.intToSdfPath(int(value)))
        def on_contact(headers, data):
            try:
                for header in headers:
                    for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                        item = data[index]
                        normal = [float(v) for v in item.normal]
                        raw = item.impulse
                        try:
                            impulse = [float(v) for v in raw]
                        except TypeError:
                            impulse = [float(raw) * v for v in normal]
                        event_rows.append({
                            "source": "subscribe_contact_report_events", "phase": "POST",
                            "step": step[0], "interval_dt_s": dt,
                            "actor0": sdf(header.actor0), "actor1": sdf(header.actor1),
                            "collider0": sdf(getattr(header, "collider0", header.actor0)),
                            "collider1": sdf(getattr(header, "collider1", header.actor1)),
                            "point_m": [float(v) for v in item.position], "normal_actor0_to_actor1": normal,
                            "impulse_on_actor0_Ns": impulse, "separation_m": float(item.separation),
                        })
            except Exception as exc:
                event_errors.append(f"{type(exc).__name__}: {exc}")
        event_subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

        sim = SimulationContext(SimulationCfg(dt=dt, render_interval=4, device="cuda:0"))
        sim.reset()
        physics_view = sim.physics_manager.get_physics_sim_view()
        body_view = physics_view.create_rigid_body_view(["/World/Probe/sensor"])
        contact_view = physics_view.create_rigid_contact_view(
            ["/World/Probe/sensor"], filter_patterns=[["/World/Probe/target"]],
            max_contact_data_count=64,
        )
        if body_view.count != 1 or contact_view.sensor_count != 1 or contact_view.filter_count != 1:
            raise RuntimeError(
                f"unexpected contact view counts body={body_view.count}, "
                f"sensor={contact_view.sensor_count}, filter={contact_view.filter_count}"
            )

        tensor_rows = []
        body_rows = []
        for _ in range(180):
            sim.step(render=False)
            pose = wp.to_torch(body_view.get_transforms()).detach().cpu().tolist()[0]
            twist = wp.to_torch(body_view.get_velocities()).detach().cpu().tolist()[0]
            body_rows.append({"step": step[0] - 1, "pose": pose, "twist": twist})
            forces, points, normals, separations, counts, starts = contact_view.get_contact_data(dt=dt)
            force_rows = wp.to_torch(forces).detach().cpu().tolist()
            point_rows = wp.to_torch(points).detach().cpu().tolist()
            normal_rows = wp.to_torch(normals).detach().cpu().tolist()
            separation_rows = wp.to_torch(separations).detach().cpu().reshape(-1).tolist()
            count_rows = wp.to_torch(counts).detach().cpu().reshape(-1).tolist()
            start_rows = wp.to_torch(starts).detach().cpu().reshape(-1).tolist()
            if len(count_rows) != 1 or len(start_rows) != 1:
                raise RuntimeError(f"unexpected tensor contact index shapes {len(count_rows)}, {len(start_rows)}")
            for k in range(int(start_rows[0]), int(start_rows[0]) + int(count_rows[0])):
                # The pinned tensor API returns a scalar normal force for this
                # backend (shape N x 1), not a world-space vec3.  Preserve the
                # native scalar and derive the requested vector from the native
                # contact normal instead of indexing it as a three-vector.
                raw_force = force_rows[k]
                if isinstance(raw_force, (int, float)):
                    normal_force = float(raw_force)
                elif len(raw_force) == 1:
                    normal_force = float(raw_force[0])
                elif len(raw_force) == 3:
                    normal_force = None
                else:
                    raise RuntimeError(f"unexpected tensor force row {raw_force!r}")
                normal = [float(v) for v in normal_rows[k]]
                force = ([normal_force * v for v in normal]
                         if normal_force is not None else [float(v) for v in raw_force])
                impulse = [v * dt for v in force]
                if not all(math.isfinite(v) for v in force + impulse):
                    raise RuntimeError("non-finite tensor contact force/impulse")
                tensor_rows.append({
                    "source": "physics_sim_view.create_rigid_contact_view.get_contact_data",
                    "phase": "POST", "step": step[0] - 1, "interval_dt_s": dt,
                    "sensor_actor": "/World/Probe/sensor", "filter_actor": "/World/Probe/target",
                    "point_m": [float(v) for v in point_rows[k]],
                    "normal_sensor_to_filter": normal,
                    "native_normal_force_N": normal_force,
                    "force_on_sensor_N": force, "impulse_on_sensor_Ns": impulse,
                    "separation_m": float(separation_rows[k]),
                })

        if callback_errors or event_errors:
            raise RuntimeError(f"native callback exception: post={callback_errors}, contact={event_errors}")
        if not tensor_rows:
            raise RuntimeError("tensor contact view produced no contact rows for the known falling-body probe")
        raw_path = ROOT / "measurements/raw-component/contact-observation-probe.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as stream:
            for row in body_rows + event_rows + tensor_rows:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        max_impulse = max(math.sqrt(sum(v*v for v in row["impulse_on_sensor_Ns"])) for row in tensor_rows)
        report = {
            "status": "complete", "scope": "contact-observation API probe only; not component certification",
            "checks": [
                {"name": "known-pair-tensor-contact-row-count-minimum", "measured": len(tensor_rows), "operator": ">=", "threshold": 1, "unit": "rows"},
                {"name": "postphysics-callback-exception-count-maximum", "measured": 0, "operator": "<=", "threshold": 0, "unit": "exceptions"},
                {"name": "known-pair-maximum-observed-impulse-minimum", "measured": max_impulse, "operator": ">", "threshold": 0, "unit": "N*s"},
            ],
            "provenance": {
                "event_api": "omni.physx.get_physx_simulation_interface().subscribe_contact_report_events",
                "tensor_api": "PhysxManager physics_sim_view.create_rigid_contact_view/get_contact_data(dt)",
                "callback_api": "PhysxManager.register_callback(IsaacEvents.POST_PHYSICS_STEP)",
                "device": "cuda:0", "physics_hz": 240, "actor_order": "sensor then filter",
                "event_row_count": len(event_rows), "tensor_row_count": len(tensor_rows),
            },
            "artifacts": [str(args.output.relative_to(ROOT)), str(raw_path.relative_to(ROOT))],
        }
        PhysxManager.deregister_callback(post_handle)
        post_handle = None
        event_subscription = None
        dump(args.output, report)
        print(json.dumps(report, allow_nan=False))
        return 0
    except Exception as exc:
        report = {
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "checks": [],
            "artifacts": ["measurements/raw-component/contact-observation-probe.jsonl"],
        }
        dump(args.output, report)
        print(json.dumps(report, allow_nan=False))
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
