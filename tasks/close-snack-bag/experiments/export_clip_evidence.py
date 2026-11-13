"""Keep compact measured evidence; leave full 60 Hz recordings on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
for name in ('result.json', 'trace.json', 'clip-shapes.json', 'preflight.json',
             'arguments.json', 'geometry-counts.npz', 'hold-check.json'):
    if (a.run/name).is_file():
        shutil.copy2(a.run/name, a.output/name)
source_hashes = {}
if (a.run/'sources').is_dir():
    shutil.copytree(a.run/'sources',a.output/'sources')
    for source in sorted((a.run/'sources').iterdir()):
        source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
data = np.load(a.run/'pouch-snapshots.npz')
frames = sorted(int(k[2:]) for k in data.files if k.startswith('q_'))
selected = sorted({frames[0], frames[-1]} | {frame for frame in frames if frame % 60 == 0})
compact = {'faces': data['faces']}
for key in ('edges', 'rest_initial', 'rest_final'):
    if key in data:
        compact[key] = data[key]
for frame in selected:
    for prefix in ('q_', 'b_'):
        key = f'{prefix}{frame:05d}'
        if key in data:
            compact[key] = data[key]
np.savez_compressed(a.output/'selected-states.npz', **compact)
from check_native_clip import inspect_state, ground_clearance
shapes = json.loads((a.run/'clip-shapes.json').read_text())
checks = [{'control_frame': frame, 'boxes': inspect_state(compact[f'q_{frame:05d}'],
    compact['faces'], compact[f'b_{frame:05d}'], shapes),
    'ground':ground_clearance(compact[f'q_{frame:05d}'],compact[f'b_{frame:05d}'],shapes)} for frame in selected]
(a.output/'selected-box-checks.json').write_text(json.dumps({
    'scope': 'CPU SAT cross-check of selected states only; whole-episode GPU counts are separate',
    'states_with_intrusion': sum(any(box['penetrating_triangles'] for box in row['boxes']) for row in checks),
    'states_with_ground_intrusion':sum(row['ground']['intrusion'] for row in checks),
    'rows': checks}, indent=2))
metadata = {'raw_run': str(a.run), 'selected_control_frames': selected,
            'frozen_source_sha256': source_hashes,
            'missing_frozen_source_note': None if source_hashes else
                'This earlier diagnostic did not freeze source files at reset; settings and current source must be compared explicitly.',
            'raw_full_recording_kept_on_server': True}
asset_root = Path(__file__).resolve().parents[1]/'assets/clip_v5'
frozen_tool=a.run/'sources/clip_compliant_v1.xml'
if frozen_tool.exists():
    metadata['clip_asset_sha256']={'experiments/assets/clip_compliant_v1.xml':hashlib.sha256(frozen_tool.read_bytes()).hexdigest()}
else:
    metadata['clip_asset_sha256'] = {str(file.relative_to(asset_root)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(asset_root.rglob('*')) if file.is_file()}
(a.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
print(a.output)
