"""CPU shape properties; no simulator dependency and no native-pass assertion."""

import math

from fixture_seed import fixture_parts
from geometry_seed import CONTACT_OFFSET


def test_jaw_compounds_are_connected_solids_instead_of_floating_pads():
    for name in ("left_jaw", "right_jaw"):
        pieces = fixture_parts()[name]["pieces"]
        visited = {0}
        while True:
            previous = set(visited)
            for i, a in enumerate(pieces):
                for j, b in enumerate(pieces):
                    overlap = all(
                        min(a["center"][k] + a["size"][k] / 2, b["center"][k] + b["size"][k] / 2)
                        - max(a["center"][k] - a["size"][k] / 2, b["center"][k] - b["size"][k] / 2)
                        > 1e-9
                        for k in range(3)
                    )
                    if i in visited and overlap:
                        visited.add(j)
            if previous == visited:
                break
        assert len(visited) == len(pieces)


def test_cover_and_barrel_leave_clearance_for_the_actual_mating_bodies():
    frame = fixture_parts()["frame"]["pieces"]
    center = (0, -0.018)
    for prefix, moving_radius in (("Cover", 0.0039), ("Barrel", 0.008)):
        sectors = [p for p in frame if p["name"].startswith(prefix)]
        assert len(sectors) == 24
        for piece in sectors:
            a, b = piece["footprint"][2:4]
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            distance = (
                abs((b[0] - a[0]) * (center[1] - a[1]) - (b[1] - a[1]) * (center[0] - a[0]))
                / length
            )
            assert distance - moving_radius > 2 * CONTACT_OFFSET
