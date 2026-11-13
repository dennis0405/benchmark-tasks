"""Instrument upstream bag drive; keep physics and robot gesture unchanged."""
import json, time, importlib.metadata as metadata, sys
from pathlib import Path
import numpy as np
from tests import test_open_a_bag as upstream
from tests.fixtures import grasp
from robosimstudio_materials.bag import layer_gap

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
meta = {"python": sys.version, "versions": {n: metadata.version(n) for n in
        ("newton", "warp-lang", "superdex-physics", "mujoco")}, "device": "cpu",
        "upstream_commit": "4e52e1a76437ef1296028afea744437f6e9469c0"}
(out / "environment.json").write_text(json.dumps(meta, indent=2))
for on in (True, False):
    name = "self-contact-on" if on else "self-contact-off"
    samples, snapshots, faces = [], {}, None
    original = grasp.tick
    def tick(session, steps):
        global faces
        for _ in range(steps):
            original(session, 1)
            sim = session.env
            bag = sim["bag"]
            q = np.asarray(bag.particle_q, dtype=np.float64)
            row = {"step": int(sim.step_count), "gap_m": float(layer_gap(q, bag.entity)),
                   "finite": bool(np.isfinite(q).all())}
            samples.append(row)
            if len(samples) == 1 or len(samples) % 15 == 0:
                snapshots[f"q_{len(samples):05d}"] = q.copy()
            if faces is None:
                faces = np.asarray(bag.entity.surface()[1])
    grasp.tick = upstream.tick = tick
    print("BEGIN", name, flush=True)
    started = time.monotonic()
    try:
        got = upstream._drive(19761 if on else 19762, self_contact=on)
        result = {k: v for k, v in got.items() if k not in ("sim", "spec")}
        result.update(self_contact=on, wall_seconds=time.monotonic()-started,
                      sampled_frames=len(samples), min_recorded_gap_m=min(x["gap_m"] for x in samples),
                      all_recorded_finite=all(x["finite"] for x in samples))
        (out / (name+".json")).write_text(json.dumps(result, indent=2))
        print("RESULT", name, json.dumps(result), flush=True)
    finally:
        grasp.tick = upstream.tick = original
        (out / (name+"-trace.json")).write_text(json.dumps(samples))
        if snapshots:
            np.savez_compressed(out / (name+"-snapshots.npz"), faces=faces, **snapshots)
