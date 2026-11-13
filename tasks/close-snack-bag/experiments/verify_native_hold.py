"""Strict three-second closing-only diagnostic gate, separate from task scorer.

The pouch rests on a table; this does not establish an airtight seal or lift
capacity. Thresholds are declared here, and geometry must pass for the entire
recorded episode, including initial closure.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def evaluate(result, trace, shapes, *, seconds=3., slip_limit=.002):
    report = {'required_hold_seconds': seconds, 'slip_limit_m': slip_limit,
              'contact_span_min_m': .08, 'scope': 'stationary table-supported closing only'}
    geometry = result.get('geometry', {})
    report['complete_physics_cadence'] = bool(geometry and
        geometry.get('states_checked') == len(trace)*result.get('substeps', 0)+1 and result.get('substeps', 0) > 0)
    report['all_step_geometry_passed'] = bool(report['complete_physics_cadence'] and
        geometry['clip_intrusion_states'] == 0 and geometry['self_crossing_states'] == 0)
    report['all_finite'] = bool(result['all_finite'])
    jaw_shapes = [str(i) for i, (owner, local, half) in enumerate(zip(
        shapes['shape_body'], shapes['shape_local'], shapes['shape_scale']))
        if owner >= 0 and local[0] > 0 and half[1] >= .045]
    report['jaw_shape_ids'] = jaw_shapes
    n = int(round(seconds*60))+1
    windows = []
    if len(jaw_shapes) == 2:
        for start in range(len(trace)-n+1):
            window = trace[start:start+n]
            if any('shape_fold_contacts' not in row for row in window):
                continue
            contact = all(all(row['shape_fold_contacts'][sid]['active_fold_contacts'] > 0 and
                row['shape_fold_contacts'][sid]['fold_contact_y_span_m'] >= .08
                for sid in jaw_shapes) for row in window)
            if not contact:
                continue
            fold = np.array([row['fold_centroid_clip_local_m'] for row in window])
            root = np.array([row['clip_root_pose'][:3] for row in window])
            jaw = np.array([row['jaw_angle_rad'] for row in window])
            slip = float(np.linalg.norm(fold-fold[0], axis=1).max())
            drift = float(np.linalg.norm(root-root[0], axis=1).max())
            if slip > slip_limit or drift > .002 or jaw.min() < -.56 or jaw.max() > .03 or np.ptp(jaw) > .02:
                continue
            windows.append({'start_s': window[0]['step']/60, 'end_s': window[-1]['step']/60,
                'hold_seconds': (window[-1]['step']-window[0]['step'])/60,
                'max_relative_slip_m': slip, 'max_root_drift_m': drift,
                'jaw_range_rad': [float(jaw.min()), float(jaw.max())],
                'minimum_contact_span_m': min(row['shape_fold_contacts'][sid]['fold_contact_y_span_m']
                    for row in window for sid in jaw_shapes)})
    report['qualified_window_count'] = len(windows)
    report['best_window'] = min(windows, key=lambda w: w['max_relative_slip_m']) if windows else None
    report['passed'] = bool(report['all_step_geometry_passed'] and report['all_finite'] and windows)
    report['limitations'] = ['No coplanar-overlap or continuous inter-step collision certificate',
        'Pre-folded reset and open clip already around mouth; no insertion or YAM',
        'Uncalibrated material and force telemetry; no air, contents, lift or airtightness test']
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    report = evaluate(*(json.loads((args.run/name).read_text()) for name in
        ('result.json', 'trace.json', 'clip-shapes.json')))
    (args.run/'hold-check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
