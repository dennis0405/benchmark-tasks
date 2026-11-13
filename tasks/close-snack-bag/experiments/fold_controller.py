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
        panels={arm:self.session.arms[arm] for arm in targets}
        starts={arm:np.array(panel.cursor_pos) for arm,panel in panels.items()}
        turns={arm:np.array(panel.cursor_quat) for arm,panel in panels.items()}
        hz=float(self.sim.request.physics.control_hz)
        def angle(a,b):
            return 2*np.arccos(np.clip(abs(np.dot(a,b)),0.,1.))
        # Include rotation in path duration: the reference cursor helper ends
        # on position alone and can request a 90-degree turn in one frame.
        seconds=max(max(np.linalg.norm(targets[a]-starts[a])/speed,
                        angle(turns[a],quats[a])/np.deg2rad(45)) for a in panels)
        count=max(1,int(np.ceil(seconds*hz)))
        for k in range(1,count+1):
            t=k/count
            for arm,panel in panels.items():
                panel.aim(starts[arm]+t*(targets[arm]-starts[arm]),
                          self.takes._slerp(turns[arm],quats[arm],t),moving=k<count)
            yield
        for _ in range(self.frames(settle)):
            yield
            poses={a:p.ik.ee_pose(self.sim.state) for a,p in panels.items()}
            if all(np.linalg.norm(poses[a][0]-targets[a])<.003 and
                   angle(poses[a][1],quats[a])<.05 for a in panels):
                break

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
        for request in np.linspace(.9053 if ready else .79,1.,self.frames(60)):
            for panel in self.session.arms.values():
                panel.grip_request=float(request)
            yield
        yield from self.wait(30)
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
        # The pad grips a 12 mm strip, not a single point at its centre.
        # Keeping the wrist near 90 degrees leaves that strip upright; the
        # remaining free flap is shorter than the centre-to-crease radius.
        run = min(.014,np.sqrt(max(radius**2-.010**2, 1e-6)))
        points = {a:(line+run, ys[a], self.layout["board_top"]+.010) for a in hands}
        reached = False
        for _ in self.move(points, quats, .015, 100):
            q = self.sim.state.particle_q.numpy()[entity.particles]
            fold = fold_metrics(q, entity.grid, [(34, 28, 40)])["folds"][0]
            # This is a pre-press handoff trigger, not the final fold score.
            if fold["median_angle_rad"] >= 2.2 and fold["folded_fraction"] >= .6:
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
            self.failure = "pre-press fold did not reach a 2.2 rad median and 60 percent folded columns"
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
        if self.after_fold in ("regrasp", "seal"):
            from clip_controller import regrasp_fold
            held=yield from regrasp_fold(self, holder, presser)
            if held and self.after_fold=="seal":
                from clip_controller import seal_fold
                yield from seal_fold(self)
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
        yield from self.move({arm:point}, {arm:quat}, .10, 80)

    def lift_press(self, arm):
        self.phase = "lift_press_"+arm
        panel = self.session.arms[arm]
        tcp, quat = panel.ik.ee_pose(self.sim.state)
        yield from self.takes._go_both(self.film, {arm:tcp+[0,0,.08]}, {arm:quat}, .10, self.frames(80), .003)

    def press(self, arm, *, leave_down=False, near=.040, along_fold=False):
        self.phase = "press_"+arm
        panel = self.session.arms[arm]
        sign = 1. if arm=="left" else -1.
        quat = self.takes._fold_hand(arm)
        panel.grip_request = .6
        top, line, yc = (self.layout[k] for k in ("board_top", "line", "center_y"))
        yield from self.move({arm:(line+.010, yc+sign*.14, top+.12)}, {arm:quat}, .06, 100)
        jaws = self.takes._fold_jaws(self.film, arm)
        from scipy.spatial.transform import Rotation
        actual_quat=panel.ik.ee_pose(self.sim.state)[1]
        _,corners,_=self.takes._fold_pads(self.film,arm)
        # Predict the pad envelope in the TARGET wrist orientation. Using
        # the current lowest corner mixed two wrist frames and moved the
        # press into the other hand as the wrist finished turning.
        delta=Rotation.from_quat(quat)*Rotation.from_quat(actual_quat).inv()
        lowest=float(delta.apply(corners-jaws)[:,2].min())
        # The first side press leaves the other rim grip unobstructed.
        # After that raw rim is released, sweep a centre pad along X over
        # the remaining upright strip, instead of leaving a central gap.
        point=np.array([line+(.002 if along_fold else .012),yc+sign*near,
                        top+(.0031 if along_fold else .0035)-lowest])
        yield from self.move({arm:point+[0,0,.0215]},{arm:quat},.12,60)
        yield from self.move({arm:point},{arm:quat},.025,30)
        yield from self.wait(20)
        # A modest compliant push, then a stroke along the fold. It acts
        # through the free finger slide; no film point is prescribed.
        start = np.asarray(panel.cursor_pos).copy()+[0,0,-.001]
        for t in np.linspace(0, 1, self.frames(180)):
            stroke=[.024*t,0,0] if along_fold else [0,sign*.04*t,0]
            panel.aim(start + stroke, quat, moving=True)
            yield
        if along_fold:
            yield from self.wait(30)
        if not leave_down:
            yield from self.lift_press(arm)
