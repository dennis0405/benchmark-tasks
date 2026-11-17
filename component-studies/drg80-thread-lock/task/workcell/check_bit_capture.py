"""Probe a free driver against the fixed socket or the full threaded workcell."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--hz", type=int, choices=(240, 480), default=240)
p.add_argument("--continuous-socket", action="store_true")
p.add_argument("--six-faces", action="store_true")
p.add_argument(
    "--mounted-offset-mm",
    type=float,
    help="use the actual threaded workcell and a single chamfered driver",
)
p.add_argument("--compliant-feed", action="store_true")
p.add_argument("--long-lead", action="store_true")
a = p.parse_args()
app = AppLauncher(headless=True).app

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxManager, PhysxCfg
from build_bench import build, continuous_ring
from native_bench import Bench
from hex_fit import six_socket_faces
from hex_fit import author_bit, CHAMFER, LONG_CHAMFER, hex_fit


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / a.hz,
            device="cuda:0",
            physics=PhysxCfg(
                solver_type=0, enable_external_forces_every_iteration=False
            ),
        )
    )
    stage = sim.stage
    mat = UsdShade.Material.Define(stage, "/World/Steel")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.45)
    api.CreateDynamicFrictionAttr(0.3)
    api.CreateRestitutionAttr(0.0)
    cases = []
    mounted = a.mounted_offset_mm is not None
    if mounted:
        fixture = a.out.with_suffix(".usda").resolve()
        build(fixture, a.assets, continuous_seat=True, implicit_axial=True)
        UsdGeom.Xform.Define(
            stage, "/World/Press"
        ).GetPrim().GetReferences().AddReference(str(fixture))
    bevel_size = LONG_CHAMFER if a.long_lead else CHAMFER
    for bevel in (bevel_size,) if mounted else (0.0, bevel_size):
        for dx in [a.mounted_offset_mm / 1000] if mounted else [-0.00040, 0.0, 0.00040]:
            i = len(cases)
            center = np.array(
                [0.36, -0.018, 0.8517] if mounted else [0.2 * i, 0.0, 0.75]
            )
            offset = np.array([0.0, dx, 0.0])
            for shape, file, pos in (
                ("Socket", "socket_adjuster", center),
                ("Driver", "hex_driver", center + offset + [0, 0, 0.0015]),
            ):
                if mounted and shape == "Socket":
                    continue
                path = f"/World/{shape}{i}"
                item = UsdGeom.Xform.Define(stage, path)
                item.GetPrim().GetReferences().AddReference(
                    str((a.assets / "tools" / (file + ".usda")).resolve())
                )
                item.AddTranslateOp().Set(Gf.Vec3d(*pos))
                if shape == "Socket":
                    if a.six_faces:
                        six_socket_faces(stage, path)
                    if a.continuous_socket:
                        continuous_ring(stage, path, "SocketWall")
                    UsdPhysics.RigidBodyAPI(item.GetPrim()).CreateKinematicEnabledAttr(
                        True
                    )
                else:
                    author_bit(
                        UsdGeom.Mesh(stage.GetPrimAtPath(path + "/HexTip")), bevel
                    )
                PhysxSchema.PhysxContactReportAPI.Apply(
                    item.GetPrim()
                ).CreateThresholdAttr(0.0)
                UsdShade.MaterialBindingAPI.Apply(item.GetPrim()).Bind(
                    mat,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
            cases.append((bevel, center, offset))
    sim.reset()
    world = PhysxManager.get_physics_sim_view()
    bodies = world.create_rigid_body_view(
        [f"/World/Driver{i}" for i in range(len(cases))]
    )
    idx = wp.array(np.arange(len(cases), dtype=np.uint32), device="cuda:0")
    order = [int(str(path).rsplit("Driver", 1)[1]) for path in bodies.prim_paths]
    assert sorted(order) == list(range(len(cases)))
    cases = [cases[i] for i in order]
    bench = (
        Bench(
            "/World/Press",
            a.hz,
            tools=["/World/Driver0"],
            native_friction=True,
            implicit_axial=True,
            transmitted_reaction=True,
            rise_seconds=0.16,
        )
        if mounted
        else None
    )
    pairs = [
        world.create_rigid_contact_view(
            [f"/World/Driver{i}"],
            filter_patterns=[
                ["/World/Press/screw" if mounted else f"/World/Socket{i}"]
            ],
            max_contact_data_count=128,
        )
        for i in order
    ]
    init = np.array([c + o for _, c, o in cases])
    peak = np.zeros(len(cases))
    max_pen = np.zeros(len(cases))
    captured = np.zeros(len(cases), int)
    measured = []
    rows = []
    for tick in range(2 * a.hz):
        pose = bodies.get_transforms().numpy()
        vel = bodies.get_velocities().numpy()
        R = Rotation.from_quat(pose[:, 3:])
        # Support most of the weight. The tether's lateral equilibrium is the
        # MISALIGNED start and never receives a socket-centre target.
        force = np.zeros((len(cases), 3))
        force[:, :2] = (20.0 if a.compliant_feed else 80.0) * (
            init[:, :2] - pose[:, :2]
        ) - 2.0 * vel[:, :2]
        force[:, 2] = 0.09 * 9.81 + np.clip(
            (30.0 if a.compliant_feed else 10.0) * (-0.008 - vel[:, 2]), -0.35, 0.35
        )
        torque = np.clip(
            -(0.6 if a.compliant_feed else 0.02) * R.as_rotvec()
            - (0.009 if a.compliant_feed else 0.002) * vel[:, 3:],
            -0.02,
            0.02,
        )
        bodies.apply_forces_and_torques_at_position(
            wp.array(force.astype(np.float32), device="cuda:0"),
            wp.array(torque.astype(np.float32), device="cuda:0"),
            None,
            idx,
            True,
        )
        if bench is not None:
            bench.step()
        sim.step(render=False)
        pose = bodies.get_transforms().numpy()
        measured = []
        for i, ((ch, center, offset), view) in enumerate(zip(cases, pairs)):
            sr = np.eye(3)
            if bench is not None:
                socket = bench.bodies["screw"].get_transforms().numpy()[0]
                center, sr = socket[:3], Rotation.from_quat(socket[3:]).as_matrix()
            fit = hex_fit(
                pose[i, :3],
                Rotation.from_quat(pose[i, 3:]).as_matrix(),
                center,
                sr,
                ch,
            )
            force_d, _, _, sep, ct, first = view.get_contact_data(dt=1 / a.hz)
            n = int(ct.numpy().ravel()[0])
            j = int(first.numpy().ravel()[0])
            load = 0.0
            if n:
                load = float(np.abs(force_d.numpy().ravel()[j : j + n]).sum())
                max_pen[i] = max(
                    max_pen[i], float(-sep.numpy().ravel()[j : j + n].min())
                )
            peak[i] = max(peak[i], load)
            good = (
                fit["straight_flank_depth_m"] >= 0.0015
                and fit["wall_clearance_m"] is not None
                and fit["wall_clearance_m"] >= -0.00005
                and fit["floor_clearance_m"] >= -0.00005
            )
            captured[i] = captured[i] + 1 if good else 0
            measured.append(
                dict(
                    chamfer_m=ch,
                    initial_offset_m=float(offset[1]),
                    lateral_offset_m=float(np.linalg.norm(pose[i, :2] - center[:2])),
                    **fit,
                )
            )
        if tick % (a.hz // 20) == 0:
            rows.append(measured)
    out = [
        dict(
            **m,
            pass_capture=bool(n >= a.hz // 4 and pen < 0.002 and f < 250),
            max_normal_n=float(f),
            max_native_penetration_m=float(pen),
        )
        for m, n, f, pen in zip(measured, captured, peak, max_pen)
    ]
    a.out.write_text(
        json.dumps(
            dict(
                rate_hz=a.hz,
                continuous_socket=a.continuous_socket,
                six_faces=a.six_faces,
                mounted_offset_mm=a.mounted_offset_mm,
                final_fixture=bench.read() if mounted else None,
                compliant_feed=a.compliant_feed,
                scope="contact fixture, not robot points",
                cases=out,
            ),
            indent=2,
        )
    )
    a.out.with_suffix(".trace.json").write_text(json.dumps(rows))
    print(json.dumps(out), flush=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
