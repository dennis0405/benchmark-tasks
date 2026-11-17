"""Instrumented commissioning of the original workcell. Direct tool wrenches, no YAM points."""

import argparse
import json
import math
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--rise-seconds", type=float, default=0.0)
p.add_argument("--native-friction", action="store_true")
p.add_argument("--continuous-seat", action="store_true")
p.add_argument("--velocity-iters", type=int, default=4)
p.add_argument("--valve-force", type=float, default=5.0)
p.add_argument("--tightening-travel", type=float, default=1.0)
p.add_argument("--drive-instrument", action="store_true")
p.add_argument("--release-only", action="store_true")
p.add_argument("--early", action="store_true")
p.add_argument("--hz", type=int, default=240)
args = p.parse_args()
app = AppLauncher(headless=True).app
import numpy as np
import isaaclab.sim as sim_utils
from pxr import UsdGeom
from build_bench import build
from native_bench import Bench


def main():
    here = args.out.parent
    here.mkdir(parents=True, exist_ok=True)
    local = here / (args.out.stem + ".usda")
    build(
        local,
        args.assets,
        instrumented=args.drive_instrument,
        velocity_iters=args.velocity_iters,
        continuous_seat=args.continuous_seat,
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / args.hz, device="cuda:0", render_interval=12)
    )
    x = UsdGeom.Xform.Define(sim.stage, "/World/Press")
    x.GetPrim().GetReferences().AddReference(str(local.resolve()))
    sim.reset()
    bench = Bench(
        "/World/Press",
        args.hz,
        valve_force=args.valve_force,
        rise_seconds=args.rise_seconds,
        native_friction=args.native_friction,
    )

    def tool_instrument(q, phase, t):
        import warp as wp

        def send(setter, values, device="cpu"):
            indices = (
                bench.ids
                if device == bench.device
                else wp.array([0], dtype=wp.uint32, device="cpu")
            )
            setter(
                wp.array(np.array([values], dtype=np.float32), device=device), indices
            )

        if "limits" not in tool_instrument.__dict__:
            tool_instrument.limits = bench.art.get_dof_max_forces().numpy()[0].copy()
            tool_instrument.stiff = bench.art.get_dof_stiffnesses().numpy()[0].copy()
            tool_instrument.damp = bench.art.get_dof_dampings().numpy()[0].copy()
            tool_instrument.phase = None
        engaged = phase in ("adjust", "lock")
        if phase != tool_instrument.phase:
            k = tool_instrument.stiff.copy()
            b = tool_instrument.damp.copy()
            mx = tool_instrument.limits.copy()
            for a, gain, limit in ((1, 0.50, 0.08), (3, 0.15, 0.05)):
                j = bench.index[a]
                enable = engaged and (a == 1 or phase == "lock")
                k[j] = gain if enable else 0.0
                b[j] = 0.002 if enable else 0.0
                mx[j] = limit
            send(bench.art.set_dof_max_forces, mx)
            send(bench.art.set_dof_stiffnesses, k)
            send(bench.art.set_dof_dampings, b)
            tool_instrument.phase = phase
            tool_instrument.start = q.copy()
        if engaged:
            targets = bench.art.get_dof_position_targets().numpy()[0].copy()
            start = tool_instrument.start
            end_st = target
            # At the nominal 90-degree setting the unloaded angular sum for first
            # bearing contact is -2*pi. Turn beyond first contact to store strain.
            if phase == "adjust":
                targets[bench.index[1]] = start[1] + min(1, t / 1.8) * (
                    end_st - start[1]
                )
            else:
                targets[bench.index[1]] = end_st
                end_nt = -math.tau - end_st - args.tightening_travel
                targets[bench.index[3]] = start[3] + min(1, t / 2.3) * (
                    end_nt - start[3]
                )
            send(bench.art.set_dof_position_targets, targets, bench.device)

    tr = []
    try:
        target = (math.radians(87) / 300 - 0.0057) * math.tau / 0.001
        for phase, duration in (
            ("reset", 0.5),
            ("adjust", 2),
            ("lock", 3),
            ("released", 1),
            ("open1", 1),
            ("close1", 1),
            ("open2", 1),
            ("close2", 1),
            ("open3", 1),
            ("close3", 1),
            ("open4", 1),
        ):
            if args.release_only and phase == "open1":
                break
            for i in range(round(duration * args.hz)):
                q, v = bench.state()
                diag = np.zeros(8)
                if phase in ("adjust", "lock"):
                    # Bounded external wrench on the physical screw, compensated only by the test driver.
                    # Slow finite-speed instrument acts as a wrench through native
                    # impulses. Use an implicit inertia denominator at this rate.
                    kp = 0.35
                    kv = 0.0006
                    effective_i = 7e-7
                    torque = (kp * (target - q[1]) - (kv + kp / args.hz) * v[1]) / (
                        1 + (kv + kp / args.hz) / (args.hz * effective_i)
                    )
                    driver = np.clip(torque, -0.022, 0.022)
                    diag[1] += driver
                if phase == "lock":
                    # A slowly increasing handle torque after the free-running approach.
                    # Speed damping is implicit for the real nut inertia.
                    torque = -(0.0015 + 0.0135 * min(1, i / (2 * args.hz)))
                    diag[1] += torque
                    diag[3] += torque
                if phase.startswith(("open", "close")):
                    lever = math.radians(30 if phase.startswith("open") else -30)
                    diag[7] = np.clip(0.2 * (lever - q[7]) - 0.004 * v[7], -0.08, 0.08)
                if args.drive_instrument:
                    diag[:4] = 0
                    tool_instrument(q, phase, i / args.hz)
                bench.step(diag)
                sim.step(render=False)
                row = bench.read()
                if i == 0 or i % 12 == 0:
                    row["poses"] = bench.poses()
                tr.append(dict(phase=phase, **row))
                if args.early and (
                    row["seat_gap"] < -0.0002
                    or abs(row["sz"]) > 0.0048
                    or (
                        row["p"] > 0.002
                        and abs(row["angle"] - (3 + math.degrees(300 * row["p"]))) > 4.0
                    )
                ):
                    raise ValueError(f"guide or seat invalid in {phase} at {i}: {row}")
                if i > args.hz // 4 and (
                    not np.all(np.isfinite(q)) or max(abs(v[:4])) > 500
                ):
                    raise ValueError(
                        f"diverging fixture speed in {phase}: {v.tolist()}"
                    )
            print(
                phase,
                {
                    k: round(tr[-1][k], 5)
                    for k in (
                        "sz",
                        "st",
                        "nz",
                        "nt",
                        "angle",
                        "seat_n",
                        "stop_n",
                        "lever",
                    )
                },
                flush=True,
            )
    finally:
        args.out.with_suffix(".trace.json").write_text(json.dumps(tr, allow_nan=False))

    def hold(phase):
        return [r for r in tr if r["phase"] == phase][-args.hz // 4 :]

    if args.release_only:
        args.out.write_text(
            json.dumps(
                dict(
                    released_min_n=min(r["seat_n"] for r in hold("released")),
                    final=hold("released")[-1],
                ),
                indent=2,
            )
        )
        return
    checks = {
        "no_reset_preload": max(r["seat_n"] for r in hold("reset")) < 0.1,
        "full_driver_adjustment": abs(hold("adjust")[-1]["st"] - target)
        < math.radians(3),
        "preload_after_release": min(r["seat_n"] for r in hold("released")) >= 2,
        "three_closed": all(
            max(r["angle"] for r in hold("close" + str(i))) <= 10 for i in range(1, 4)
        ),
        "three_retained_open": all(
            max(abs(r["angle"] - 90) for r in hold("open" + str(i))) <= 3
            for i in range(2, 5)
        ),
        "actual_stop_contact": all(
            min(r["stop_n"] for r in hold("open" + str(i))) >= 0.1 for i in range(2, 5)
        ),
    }
    from audit_workcell import audit

    qualification = audit(tr, args.hz)
    args.out.write_text(
        json.dumps(
            dict(
                checks={k: bool(v) for k, v in checks.items()},
                fixture_checks_pass=all(checks.values()),
                qualification=qualification,
                all_necessary_pass=bool(
                    all(checks.values()) and qualification["all_necessary_pass"]
                ),
                final={
                    phase: hold(phase)[-1]
                    for phase in dict.fromkeys(r["phase"] for r in tr)
                },
            ),
            indent=2,
        )
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    args.out.write_text(
        json.dumps({"error": str(exc), "all_necessary_pass": False}, indent=2)
    )
finally:
    app.close()
