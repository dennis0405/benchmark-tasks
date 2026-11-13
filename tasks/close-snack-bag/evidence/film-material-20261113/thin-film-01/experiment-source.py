"""Thin-film geometry/contact sensitivity, using the pinned upstream CPU bag."""
import json,sys,time
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio_materials.bag import CANVAS, layer_gap, mouth_opening
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
# Hypothesis only: not measured snack-package material.
film=rss.replace(CANVAS,thickness=80e-6,areal_density=0.08,shell_young=1e8,shell_poisson=0.35)
for radius,on in ((.002,True),(.001,True),(.0001,True),(.0001,False)):
    name=f"r{radius*1e6:.0f}um-{'on' if on else 'off'}"
    print("BEGIN",name,flush=True); started=time.monotonic()
    material=rss.replace(film,particle_radius=radius)
    scene=rss.Scene(cell="bench_0728",physics=rss.BAG_60HZ,name=name)
    scene.add(rss.bag(name="bag",size=(.20,.16),cells=(40,32),lip=.02,
                      material=material,self_contact=on,settle_frames=0))
    sim=scene.build(device="cpu");bag=sim["bag"]
    frames={};samples=[]
    for k in range(180):
        sim.run(1).sync()
        q=np.asarray(bag.particle_q,dtype=np.float64)
        gap=float(layer_gap(q,bag.entity))
        samples.append({"step":k+1,"gap_m":gap if np.isfinite(gap) else None,
                        "finite":bool(np.isfinite(q).all())})
        if k%15==0 or k==179:frames[f"q_{k+1:05d}"]=q.copy()
    result={"name":name,"physical_thickness_m":80e-6,"areal_density_kg_m2":.08,
      "assumed_young_pa":1e8,"contact_radius_m":radius,"self_contact":on,
      "mesh_cells":[40,32],"steps":180,"control_hz":60,"wall_seconds":time.monotonic()-started,
      "min_recorded_gap_m":min(x["gap_m"] for x in samples if x["gap_m"] is not None),
      "finite":all(x["finite"] for x in samples),"mouth":mouth_opening(q,bag.entity),
      "limitation":"unforced rest test; no plasticity, clip or calibrated material"}
    (out/(name+".json")).write_text(json.dumps(result,indent=2))
    (out/(name+"-trace.json")).write_text(json.dumps(samples))
    np.savez_compressed(out/(name+"-snapshots.npz"),faces=bag.entity.surface()[1],**frames)
    print("RESULT",json.dumps(result),flush=True)
    sim.close() if hasattr(sim,"close") else None
