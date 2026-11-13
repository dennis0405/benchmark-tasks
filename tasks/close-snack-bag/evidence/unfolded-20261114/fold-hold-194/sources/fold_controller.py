"""Contact-driven pouch folding commands, using the pinned RSS YAM IK panel.

The upstream paper take supplies pad-frame geometry and smooth IK cursor
motion. This controller only writes robot cursor/grip requests.
"""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from fold_metrics import fold_metrics


class FoldController:
    def __init__(self, sim, session):
        sys.path.insert(0, str(Path(__import__("robosimstudio").__file__).resolve().parents[1] / "scripts"))
        from recording import takes
        self.takes = takes
        self.sim, self.session = sim, session
        self.film = SimpleNamespace(sim=sim, session=session)
        self.phase = "settle"
        self.marks = {}
        self.failure = None
        self.after_fold = "press"

    def wait(self, frames):
        yield from range(self.frames(frames))

    def frames(self, at_60_hz):
        return max(1, round(at_60_hz*self.sim.request.physics.control_hz/60))

    def move(self, points, quats, speed=.05, settle=100):
        targets = self.takes._fold_targets(self.film, points, quats)
        yield from self.takes._go_both(self.film, targets, quats, speed, self.frames(settle), .003)

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
        # A 30 mm fold about the board's front edge, within the empty neck.
        line = mouth + .030
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
        if not ready:
            self.phase = "lift_to_board"
            points = {a: self.takes._fold_jaws(self.film, a).copy() for a in hands}
            for point in points.values():
                point[2] = max(point[2], self.layout["board_top"]+.001)
            yield from self.move(points, hands, .025, 80)
        self.phase = "fold"
        jaws = {a: self.takes._fold_jaws(self.film, a).copy() for a in hands}
        radius = max(.012, line-np.mean([p[0] for p in jaws.values()]))
        for phi in np.linspace(0, np.pi/2, self.frames(240)):
            turn = min(88., np.rad2deg(phi)*1.45)
            quats = {a: self.takes._fold_hand(a, turn) for a in hands}
            points = {a: (line-radius*np.cos(phi), ys[a], jaws[a][2]+radius*np.sin(phi)) for a in hands}
            targets = self.takes._fold_targets(self.film, points, quats)
            for arm in hands:
                self.session.arms[arm].aim(targets[arm], quats[arm], moving=True)
            yield
        # Keep the wrist below 90 degrees, as in the reference paper fold:
        # lay the upright flap over the body, release, then crease by pressing.
        self.phase = "lay_flap"
        quats = {a:self.takes._fold_hand(a, 88.) for a in hands}
        run = np.sqrt(max(radius**2-.008**2, 1e-6))
        points = {a:(line+run, ys[a], self.layout["board_top"]+.008) for a in hands}
        reached = False
        for _ in self.move(points, quats, .015, 100):
            q = self.sim.state.particle_q.numpy()[entity.particles]
            fold = fold_metrics(q, entity.grid, [(34, 28, 40)])["folds"][0]
            if fold["median_angle_rad"] >= 2.5 and fold["folded_fraction"] >= .8:
                print("FOLD_REACHED", fold, flush=True)
                reached = True
                # Stop at the measured folded state instead of driving on
                # into the board after the flap is already over.
                for panel in self.session.arms.values():
                    pos, quat = panel.ik.ee_pose(self.sim.state)
                    panel.aim(pos, quat, moving=False)
                break
            yield
        if not reached:
            self.failure = "fold did not reach 2.5 rad across 80 percent of measured columns"
            self.phase = "fold_failed"
            return
        self.marks["carried"] = self.sim.state.particle_q.numpy()[entity.particles].copy()
        if self.after_fold == "hold":
            self.phase = "fold_hold"
            yield from self.wait(180)
            return
        # Keep whichever hand still has bilateral contact; a fixed left/right
        # order dropped the flap when the left grip had slipped during the turn.
        solver = self.sim.pipeline.program.solver
        forces = solver.shape_contact_forces()
        owners = self.sim.model.shape_body.numpy()
        loads = {a:min(float(np.linalg.norm(forces[np.isin(owners, side)].sum(axis=0)))
                       for side in index.finger_bodies)
                 for a,index in self.sim.scene.embodiment.arms.items()}
        holder = max(loads, key=loads.get)
        presser = "left" if holder=="right" else "right"
        print("SUPPORT_HAND", holder, loads, flush=True)
        yield from self.retract_grasp(presser)
        yield from self.press(presser, leave_down=True)
        if self.after_fold == "regrasp":
            from clip_controller import regrasp_fold
            yield from regrasp_fold(self, holder, presser)
            return
        yield from self.retract_grasp(holder)
        yield from self.press(holder, leave_down=True)
        yield from self.lift_press("right")
        yield from self.lift_press("left")
        self.phase = "watch"
        yield from self.wait(180)

    def retract_grasp(self, arm):
        self.phase = "release_"+arm
        panel = self.session.arms[arm]
        panel.grip_request = .79
        yield from self.wait(45)
        quat = panel.ik.ee_pose(self.sim.state)[1]
        point = self.takes._fold_jaws(self.film, arm)+[0, .10 if arm=="left" else -.10, .06]
        yield from self.move({arm:point}, {arm:quat}, .05, 80)

    def lift_press(self, arm):
        self.phase = "lift_press_"+arm
        panel = self.session.arms[arm]
        tcp, quat = panel.ik.ee_pose(self.sim.state)
        yield from self.takes._go_both(self.film, {arm:tcp+[0,0,.08]}, {arm:quat}, .04, self.frames(80), .003)

    def press(self, arm, *, leave_down=False):
        self.phase = "press_"+arm
        panel = self.session.arms[arm]
        sign = 1. if arm=="left" else -1.
        quat = self.takes._fold_presser(arm)
        panel.grip_request = .6
        top, line, yc = (self.layout[k] for k in ("board_top", "line", "center_y"))
        yield from self.move({arm:(line+.010, yc+sign*.14, top+.12)}, {arm:quat}, .06, 100)
        low = self.takes._fold_low(self.film, arm)
        tcp = panel.ik.ee_pose(self.sim.state)[0]
        target = tcp + np.array([line+.012, yc+sign*.025, top+.001])-low
        yield from self.takes._go_both(self.film, {arm:target}, {arm:quat}, .025, self.frames(100), .003)
        yield from self.wait(45)
        # A modest compliant push, then a stroke along the fold. It acts
        # through the free finger slide; no film point is prescribed.
        start = target + [0, 0, -.002]
        for t in np.linspace(0, 1, self.frames(180)):
            panel.aim(start + [0, sign*.04*t, 0], quat, moving=True)
            yield
        if not leave_down:
            yield from self.lift_press(arm)
