"""CPU shape properties; no simulator dependency and no native-pass assertion."""

import math

from fixture_seed import fixture_layout, fixture_parts
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


def test_loose_nut_does_not_seat_before_the_proposed_90_degree_setting():
    layout = fixture_layout()
    screw_z = layout["adjustment_screw"][1][2]
    piston_z = layout["piston"][1][2]
    # Worst case: the loose nut co-rotates with the adjusting screw. It then
    # follows its axial travel, rather than staying at its original world Z.
    initial_stop_gap = screw_z - 0.024 - piston_z
    stroke_at_target = math.radians(90 / 2) / 150
    screw_advance = initial_stop_gap - stroke_at_target
    assert screw_advance > 0
    nut_bottom_after_adjustment = layout["nut73"][1][2] - 0.002 - screw_advance
    assert nut_bottom_after_adjustment - 0.090 > 2 * CONTACT_OFFSET + 0.0001
