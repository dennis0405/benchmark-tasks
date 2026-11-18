"""Native rigid-body release and rail clamp controls. The fixture supplies initial poses only."""
import json
import math
from pathlib import Path
import mujoco

FILE = Path(__file__).parent / "assets/tilt-gimbal/gimbal.xml"

def run(offset, degrees, hold_brake_open=False, pull=0):
    model = mujoco.MjModel.from_xml_path(str(FILE))
    data = mujoco.MjData(model)
    data.qpos[:] = [math.radians(degrees), offset, 0.012 if hold_brake_open else 0.0]
    mujoco.mj_forward(model, data)
    theta0 = data.qpos[0]
    max_drift = 0.0
    min_release = math.inf
    for _ in range(int(round(3 / model.opt.timestep))):
        if hold_brake_open:
            data.qfrc_applied[2] = 29.
        data.qfrc_applied[1] = pull
        mujoco.mj_step(model, data)
        max_drift = max(max_drift, abs(math.degrees(data.qpos[0] - theta0)))
        min_release = min(min_release, data.qpos[2])
    return dict(offset_mm=offset*1000, start_deg=degrees, end_deg=math.degrees(data.qpos[0]),
                max_drift_deg=max_drift, slide_travel_mm=1000*(data.qpos[1]-offset),
                release_mm=1000*data.qpos[2])

if __name__ == "__main__":
    rows={}
    for name, offset in [('front',.015),('center',-.019),('back',-.045)]:
        for angle in [-30,30]:
            rows[f'{name}_{angle}']=run(offset,angle)
    rows['closed_brake_pull']=run(-.019,0,pull=3)
    rows['open_brake_pull']=run(-.019,0,True,pull=3)
    print(json.dumps(rows,indent=2),flush=True)
    assert all(rows[f'center_{a}']['max_drift_deg'] <= 3 for a in (-30,30))
    assert all(rows[f'front_{a}']['end_deg'] > a+5 and rows[f'back_{a}']['end_deg'] < a-5 for a in (-30,30))
    assert abs(rows['closed_brake_pull']['slide_travel_mm']) < 1 and rows['open_brake_pull']['slide_travel_mm'] > 10
