"""Contact-driven pouch folding commands, using the pinned RSS YAM IK panel.

The upstream paper take supplies pad-frame geometry and smooth IK cursor
motion. This controller only writes robot cursor/grip requests.
"""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


class FoldController:
    def __init__(self, sim, session):
        sys.path.insert(0, str(Path(__import__("robosimstudio").__file__).resolve().parents[1] / "scripts"))
        from recording import takes
        self.takes = takes
        self.sim, self.session = sim, session
        self.film = SimpleNamespace(sim=sim, session=session)
        self.phase = "settle"
        self.marks = {}

    def wait(self, frames):
        yield from range(frames)

    def move(self, points, quats, speed=.05, settle=100):
        targets = self.takes._fold_targets(self.film, points, quats)
        yield from self.takes._go_both(self.film, targets, quats, speed, settle, .003)

    def drive(self):
        self.phase = "settle"
        ready = self.sim.snack_ready_start
        if not ready:
            yield from self.wait(60)
        entity = self.sim.scene.by_name["pouch"]
        q = self.sim.state.particle_q.numpy()[entity.particles]
        rim = q[np.unique(entity.grid[:, -1, :])]
        mouth = float(np.median(rim[:, 0]))
        yc = float(np.median(rim[:, 1]))
        z = float(np.median(rim[:, 2]))
        # One 25 mm fold, leaving 35 mm of empty neck behind it.
        line = mouth + .025
        self.layout = dict(mouth=mouth, line=line, center_y=yc,
                           board_top=float(entity.frame[2] + self.sim.snack_board_height))
        hands = {a: self.takes._fold_hand(a) for a in ("left", "right")}
        # An 80 mm pad reaches 40 mm inward from its centre. Put the centre
        # outside the pouch, so its inner 15 mm pinches the free side margin
        # and its lower finger clears the board's 120 mm width.
        ys = {"left": yc+(.055 if ready else .105), "right": yc-(.055 if ready else .105)}
        print("LAYOUT", self.layout, "rim_z", z, flush=True)
        for panel in self.session.arms.values():
            panel.grip_request = .9474 if ready else .79
        self.phase = "approach"
        if not ready:
            yield from self.move({a: (mouth+.008, yc+(.18 if a=="left" else -.18), z+.10)
                              for a in hands}, hands, .12, 120)
            yield from self.move({a: (mouth+.008, yc+(.14 if a=="left" else -.14), z)
                              for a in hands}, hands, .06, 100)
            yield from self.move({a: (mouth+.008, ys[a], z) for a in hands}, hands, .035, 100)
        self.phase = "align"
        for _ in range(0 if ready else 3):
            points={}
            q = self.sim.state.particle_q.numpy()[entity.particles]
            for arm in hands:
                jaws = self.takes._fold_jaws(self.film, arm)
                contact_xy = jaws[:2]+[0, -.035 if arm=="left" else .035]
                near = np.linalg.norm(q[:, :2]-contact_xy, axis=1)<.010
                points[arm] = jaws.copy()
                if near.any():
                    points[arm][2] = np.median(q[near, 2])
            yield from self.move(points, hands, .025, 40)
        self.phase = "pinch"
        print("PINCH_JAWS", {a:self.takes._fold_jaws(self.film,a).tolist() for a in hands}, flush=True)
        for panel in self.session.arms.values():
            panel.grip_request = 1.
        yield from self.wait(90)
        self.marks["pinched"] = self.sim.state.particle_q.numpy()[entity.particles].copy()
        self.phase = "lift_to_board"
        points = {a: self.takes._fold_jaws(self.film, a).copy() for a in hands}
        for point in points.values():
            point[2] = max(point[2], self.layout["board_top"]+.010)
        yield from self.move(points, hands, .025, 80)
        self.phase = "fold"
        jaws = {a: self.takes._fold_jaws(self.film, a).copy() for a in hands}
        radius = max(.012, line-np.mean([p[0] for p in jaws.values()]))
        for phi in np.linspace(0, np.pi, 180):
            turn = min(88., np.rad2deg(phi)*1.45)
            quats = {a: self.takes._fold_hand(a, turn) for a in hands}
            points = {a: (line-radius*np.cos(phi), ys[a], jaws[a][2]+radius*np.sin(phi)) for a in hands}
            targets = self.takes._fold_targets(self.film, points, quats)
            for arm in hands:
                self.session.arms[arm].aim(targets[arm], quats[arm], moving=True)
            yield
        yield from self.wait(60)
        self.marks["carried"] = self.sim.state.particle_q.numpy()[entity.particles].copy()
        self.phase = "release"
        for panel in self.session.arms.values():
            panel.grip_request = .79
        yield from self.wait(45)
        quats = {a: self.takes._fold_hand(a, 88.) for a in hands}
        points = {a: self.takes._fold_jaws(self.film, a)+[0, .10 if a=="left" else -.10, .06]
                  for a in hands}
        yield from self.move(points, quats, .05, 80)
        self.phase = "watch"
        yield from self.wait(180)
