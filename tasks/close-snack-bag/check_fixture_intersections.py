"""Check saved packet triangles against sampled support and press boxes."""
import argparse
import json
from pathlib import Path

import numpy as np
from check_clip_intersections import inspect_box


def main(directory):
    data = np.load(directory / 'surface-snapshots.npz')
    run = json.loads((directory / 'newton-result.json').read_text())
    fixtures = []
    if run['support_fixture']:
        fixtures.append(('support', [run['support_length']/2, run['support_width']/2, .004]))
    if run['press_fixture']:
        fixtures.append(('plate', [.035/2, run['press_width']/2, .002]))
    report = {}
    for row in run['samples']:
        points = data[f"nodes_{row['step']}"]
        report[str(row['step'])] = {
            name: inspect_box(points, data['faces'], row[f'{name}_pose_xyzw'], (0, 0, 0), half)
            for name, half in fixtures
        }
    (directory / 'fixture-intersections.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
