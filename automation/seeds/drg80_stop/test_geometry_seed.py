"""Geometric properties needed before attempting native contact tests."""

import math
from collections import Counter

from geometry_seed import CONTACT_OFFSET, export, make_assets


def inside_convex(point, polygon):
    cross = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        cross.append((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
    return all(v >= -1e-14 for v in cross)


def test_socket_and_nut_keep_their_axis_void():
    assets = make_assets()
    for name in ("thin_nut", "socket_adjuster"):
        sectors = [p for p in assets[name]["pieces"] if p["kind"] == "mesh"]
        assert len(sectors) == 24
        assert all(not inside_convex((0, 0), p["footprint"]) for p in sectors)
        for piece in sectors:
            polygon = piece["footprint"]
            center = tuple(sum(p[i] for p in polygon) / len(polygon) for i in (0, 1))
            assert inside_convex(center, polygon)


def test_each_convex_sector_has_closed_two_manifold_topology():
    for asset in make_assets().values():
        for piece in asset["pieces"]:
            if piece["kind"] != "mesh":
                continue
            edges = Counter()
            directed = Counter()
            for face in piece["faces"]:
                for a, b in zip(face, face[1:] + face[:1], strict=True):
                    assert a != b
                    edges[tuple(sorted((a, b)))] += 1
                    directed[(a, b)] += 1
            assert set(edges.values()) == {2}
            assert all(directed[(b, a)] == count for (a, b), count in directed.items())
            assert max(i for f in piece["faces"] for i in f) < len(piece["points"])
            polygon = piece["footprint"]
            for i, a in enumerate(polygon):
                b, c = polygon[(i + 1) % len(polygon)], polygon[(i + 2) % len(polygon)]
                assert (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) >= -1e-14


def test_mating_clearances_exceed_both_authored_contact_offsets():
    assets = make_assets()
    tip = assets["hex_driver"]["across_flats_m"]
    socket = assets["socket_adjuster"]["socket_across_flats_m"]
    assert 0.5 * (socket - tip) > 2 * CONTACT_OFFSET
    nut = assets["thin_nut"]
    bore_apothem = nut["bore_radius_m"] * math.cos(math.pi / 24)
    assert bore_apothem - 0.0039 > 2 * CONTACT_OFFSET
    gap = assets["open_wrench"]["jaw_gap_m"]
    assert 0.5 * (gap - nut["across_flats_m"]) > 2 * CONTACT_OFFSET


def test_export_keeps_separate_collider_sectors_and_truthful_status(tmp_path):
    manifest = export(tmp_path)
    text = (tmp_path / "thin_nut.usda").read_text()
    assert text.count('def Mesh "Sector') == 24
    assert text.count('physics:approximation = "convexHull"') == 24
    assert "native contact/dynamics NOT validated" in manifest["status"]
    assert len(manifest["files"]) == 4
