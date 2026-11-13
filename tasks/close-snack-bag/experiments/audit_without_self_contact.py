"""Negative control only: isolate self-contact's contribution to rest forces.

Never a task/demo recipe. Rigid-soft contacts remain on, self-contact is
explicitly disabled before the first step. The override is frozen in output.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import fold_pouch_robot as recorder


def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    p.add_argument("--rebase-z",action="store_true");a=p.parse_args()
    factory=recorder.make_environment
    def isolated(**kwargs):
        sim=factory(**kwargs)
        sim.pipeline.program.solver.soft.particle_enable_self_contact=False
        sim.pipeline.invalidate_graph()
        sim.snack_material_parameters["self_contact_active"]=False
        shutil.copy2(Path(__file__),a.output/"sources"/Path(__file__).name)
        (a.output/"diagnostic-override.json").write_text(json.dumps(dict(
            scope=__doc__,self_contact_enabled=False,task_success_claim=False),indent=2))
        return sim
    recorder.make_environment=isolated
    sys.argv=["fold_pouch_robot.py",str(a.output),"--ready-start","--clip","--drive",
        "--stage","rest","--frames","12","--hz","20","--physics-hz","2400",
        "--workspace","raised_flat","--material-profile","laminate","--clip-style","thin_lower",
        "--mouth-support","full","--support-extension-mm","14","--record-force-means"]
    if a.rebase_z:sys.argv.append("--rebase-z")
    recorder.main()


if __name__=="__main__":main()
