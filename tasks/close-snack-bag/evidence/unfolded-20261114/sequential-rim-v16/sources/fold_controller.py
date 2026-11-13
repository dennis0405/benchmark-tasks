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
        self.support_holds={}
        self.support_hold_enabled=False
        self.press_floor_m=.0025

    def shape_forces(self):
        average=getattr(self.sim,"snack_force_average",None)
        return average.forces() if average is not None else self.sim.pipeline.program.solver.shape_contact_forces()

    def start_support(self,arm,*,floor_z=None,quat=None,filtered=0.):
        if not self.support_hold_enabled:return
        panel=self.session.arms[arm]
        tcp,actual=panel.ik.ee_pose(self.sim.state)
        if floor_z is None:
            floor_z=self.layout["board_top"]+self.press_floor_m-float(self.takes._fold_low(self.film,arm)[2]-tcp[2])
        self.support_holds[arm]=dict(floor_z=float(floor_z),quat=list(quat if quat is not None else actual),
                                     filtered=float(filtered))

    def update_support_holds(self):
        """Keep a pressing hand in light contact while the other hand works."""
        if not self.support_holds:return
        forces=self.shape_forces()
        owners=self.sim.model.shape_body.numpy();hz=float(self.sim.request.physics.control_hz)
        for arm,hold in self.support_holds.items():
            fingers=[int(b) for side in self.sim.scene.embodiment.arms[arm].finger_bodies
                     for b in np.asarray(side).reshape(-1)]
            force=max(0.,float(forces[np.isin(owners,fingers)].sum(0)[2]))
            hold["filtered"]=.7*hold["filtered"]+.3*force
            speed=float(np.clip((hold["filtered"]-.8)*.008,-.004,.010))
            panel=self.session.arms[arm];point=np.asarray(panel.cursor_pos).copy()
            point[2]=np.clip(point[2]+speed/hz,hold["floor_z"],hold["floor_z"]+.04)
            # This is a continuous force-control gesture. Ending the drag
            # here lets the studio panel re-seat a submillimetre target on
            # the measured hand each tick, cancelling the downward integral.
            panel.aim(point,np.asarray(hold["quat"]),moving=True)

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
                          self.takes._slerp(turns[arm],quats[arm],t),moving=True)
            yield
        for _ in range(self.frames(settle)):
            yield
            poses={a:p.ik.ee_pose(self.sim.state) for a,p in panels.items()}
            if all(np.linalg.norm(poses[a][0]-targets[a])<.003 and
                   angle(poses[a][1],quats[a])<.05 for a in panels):
                break
        poses={a:p.ik.ee_pose(self.sim.state) for a,p in panels.items()}
        errors={a:dict(position_m=float(np.linalg.norm(poses[a][0]-targets[a])),
                       angle_rad=float(angle(poses[a][1],quats[a]))) for a in panels}
        reached=all(e["position_m"]<.006 and e["angle_rad"]<.08 for e in errors.values())
        for arm,panel in panels.items():panel.aim(targets[arm],quats[arm],moving=False)
        if not reached:self.marks["last_move_error"]=errors
        return reached

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
                           board_top=float(entity.frame[2] + self.sim.snack_board_height),
                           support_front=float(self.sim.snack_support_front_m))
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
            # Keep the clamped 12 mm strip tangent to the same orbit. A
            # 1.45x wrist lead lengthened the free span by about 5 percent
            # at mid-turn, imposing membrane tension instead of bending.
            turn = min(88., np.rad2deg(phi))
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
        if getattr(self,"table_insert",False) and getattr(self,"insert_mode","overhang")=="wedge":
            from clip_controller import sequential_table_fold,seal_fold
            held=yield from sequential_table_fold(self)
            if held and self.after_fold=="seal":yield from seal_fold(self)
            return
        # Keep whichever hand still has bilateral contact; a fixed left/right
        # order dropped the flap when the left grip had slipped during the turn.
        solver = self.sim.pipeline.program.solver
        forces = self.shape_forces()
        owners = self.sim.model.shape_body.numpy()
        loads = {a:min(float(np.linalg.norm(forces[np.isin(owners, side)].sum(axis=0)))
                       for side in index.finger_bodies)
                 for a,index in self.sim.scene.embodiment.arms.items()}
        holder = max(loads, key=loads.get)
        if getattr(self,"table_insert",False) and loads["right"]>.25:
            # A working right rim grip lets the left press and remain at
            # the side, avoiding an unnecessary later support transfer.
            holder="right"
        presser = "left" if holder=="right" else "right"
        print("SUPPORT_HAND", holder, loads, flush=True)
        yield from self.retract_grasp(presser)
        yield from self.press(presser, leave_down=True,
                              near=.055 if getattr(self,"table_insert",False) else .040)
        if self.failure:return
        if self.after_fold in ("regrasp", "seal"):
            if getattr(self,"table_insert",False):
                from clip_controller import prepare_table_fold,seal_fold
                held=yield from prepare_table_fold(self,holder,presser)
                if held:yield from seal_fold(self)
                return
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
        self.support_holds.pop(arm,None)
        self.phase = "release_"+arm
        panel = self.session.arms[arm]
        if getattr(self,"release_style","sliding")=="wide":
            panel.grip_request=.79
            yield from self.wait(45)
            quat=panel.ik.ee_pose(self.sim.state)[1]
            point=self.takes._fold_jaws(self.film,arm)+[0,.10 if arm=="left" else -.10,.06]
            yield from self.move({arm:point},{arm:quat},.10,80)
            return
        # After the wrist turns, opening by 20 mm spreads across the fold
        # and pries its two panels apart. Relieve only 0.8 mm per finger,
        # slide out along the long pad axis, then open and rise when clear.
        coords=self.sim.scene.embodiment.arms[arm].gripper_coords
        actual=float(self.sim.state.joint_q.numpy()[coords][0])
        panel.grip_request=float(np.clip(1-(max(0.,actual)+.0008)/.0475,0.,1.))
        yield from self.wait(15)
        quat = panel.ik.ee_pose(self.sim.state)[1]
        point = self.takes._fold_jaws(self.film, arm)+[0, .105 if arm=="left" else -.105, 0]
        yield from self.move({arm:point}, {arm:quat}, .10, 80)
        panel.grip_request=.6
        yield from self.move({arm:point+[0,0,.06]},{arm:quat},.10,80)

    def lift_press(self, arm):
        self.support_holds.pop(arm,None)
        self.phase = "lift_press_"+arm
        panel = self.session.arms[arm]
        tcp, quat = panel.ik.ee_pose(self.sim.state)
        yield from self.takes._go_both(self.film, {arm:tcp+[0,0,.08]}, {arm:quat}, .10, self.frames(80), .003)

    def press(self, arm, *, leave_down=False, near=.040, along_fold=False,sweep_end=None,at_rim=False):
        self.support_holds.pop(arm,None)
        self.phase = "press_"+arm
        panel = self.session.arms[arm]
        sign = 1. if arm=="left" else -1.
        # Keep the pinch pose's closing-axis sign. The upstream paper
        # presser reverses it, requiring an unreachable wrist branch in
        # this raised workspace. Offline FK/IK validated this 25-deg tip.
        pitch=np.deg2rad(25.)
        quat=self.takes._frame_quat((np.cos(pitch),0.,-np.sin(pitch)),(0.,0.,sign))
        panel.grip_request = .6
        top, line, yc = (self.layout[k] for k in ("board_top", "line", "center_y"))
        old_quat=panel.ik.ee_pose(self.sim.state)[1]
        current_jaw=self.takes._fold_jaws(self.film,arm)
        reached=yield from self.move({arm:current_jaw+[0,0,.10]}, {arm:old_quat}, .10, 100)
        if not reached:
            self.failure=f"{arm} pressing hover is unreachable: {self.marks.get('last_move_error')}"
            return
        jaws = self.takes._fold_jaws(self.film, arm)
        # The horizontal pinch pose puts the bulky rear capsules below its
        # pad. Use the upstream pitched pressing pose and the WHOLE finger's
        # lowest collider, not just the long pad boxes. Re-evaluate after
        # reaching that wrist pose; the approach stays well above the flap.
        from finger_geometry import low_offset
        from scipy.spatial.transform import Rotation
        tcp,actual_quat=panel.ik.ee_pose(self.sim.state)
        delta=Rotation.from_quat(quat)*Rotation.from_quat(actual_quat).inv()
        jaw_offset=delta.apply(jaws-tcp)
        lowest=low_offset(self.sim,arm,quat,panel.ik)-jaw_offset
        # The first side press leaves the other rim grip unobstructed.
        # After that raw rim is released, sweep a centre pad along X over
        # the remaining upright strip, instead of leaving a central gap.
        from stack_grasp import select_press_patch
        entity=self.sim.scene.by_name["pouch"]
        q=self.sim.state.particle_q.numpy()[entity.particles]
        patch=select_press_patch(q,entity.grid,entity.faces,yc+sign*near,
                                 self.layout.get("support_front",line))
        if at_rim:
            rim=q[np.unique(entity.grid[:,-1,:])]
            local=rim[abs(rim[:,1]-(yc+sign*near))<.018]
            if not len(local):
                self.failure="no measured free-rim points at the press entry";return
            patch["x"]=float(np.median(local[:,0])-.002)
            patch["kind"]="measured free rim, pressed vertically before creasing"
        self.marks.setdefault("press_plans",[]).append(dict(arm=arm,**patch))
        print("PRESS_PATCH",arm,patch,flush=True)
        point=np.array([patch["x"],yc+sign*near,top+.0035])-lowest
        # Complete XY travel above the highest possible upright rim. The
        # former 25 mm pre-hover entered a 27 mm flap from its side.
        reached=yield from self.move({arm:point+[0,0,.0465]},{arm:quat},.12,60)
        if not reached:
            self.failure=f"{arm} pressing approach is unreachable: {self.marks.get('last_move_error')}"
            return
        solver=self.sim.pipeline.program.solver
        owners=self.sim.model.shape_body.numpy()
        fingers={int(b) for side in self.sim.scene.embodiment.arms[arm].finger_bodies
                 for b in np.asarray(side).reshape(-1)}
        mask=np.isin(owners,list(fingers))
        def load_z():
            return max(0.,float(self.shape_forces()[mask].sum(0)[2]))
        # A contact-limited vertical approach precedes the crease stroke.
        # Only arm targets change; no particle is attached or prescribed.
        for _ in self.move({arm:point},{arm:quat},.0125,30):
            if load_z()>3.:
                actual_pos,actual_quat=panel.ik.ee_pose(self.sim.state)
                panel.aim(actual_pos,actual_quat,moving=False)
                break
            yield
        start=np.asarray(panel.cursor_pos).copy()
        floor=self.takes._fold_targets(self.film,{arm:point-[0,0,.0035-self.press_floor_m]},{arm:quat})[arm][2]
        filtered=0.;z=float(start[2]);peak=0.
        hz=float(self.sim.request.physics.control_hz)
        sweep_seconds=(max(3.,abs(sweep_end-near)/.020) if sweep_end is not None else 3.)
        count=(int(np.ceil((sweep_seconds+2.)*hz)) if sweep_end is not None else self.frames(300 if along_fold else 240))
        for k in range(count):
            force=load_z();peak=max(peak,force);filtered=.7*filtered+.3*force
            speed=float(np.clip((filtered-.8)*.008,-.004 if self.support_hold_enabled else -.006,.010))
            z=float(np.clip(z+speed/hz,floor,start[2]+.008))
            fraction=(float(np.clip((k/hz-1.)/sweep_seconds,0.,1.)) if sweep_end is not None else min(1.,k/(3.*hz)))
            # Keep the actual pressing tip on the 160 mm pouch. The former
            # side stroke ended at Y=95 mm, beyond its 80 mm half-width.
            travel=(sweep_end-near if sweep_end is not None else min(.04,max(0.,.070-near)))
            stroke=[.006*fraction if along_fold and not at_rim else 0.,sign*travel*fraction,z-start[2]]
            panel.aim(start+stroke,quat,moving=leave_down or k<count-1)
            yield
        self.marks.setdefault("press_diagnostics",[]).append(dict(
            arm=arm,centre_pass=along_fold,peak_sampled_upward_force_N=peak,
            filtered_end_force_N=filtered,final_cursor_z_m=z))
        yield from self.wait(30)
        if leave_down:
            self.start_support(arm,floor_z=floor,quat=quat,filtered=filtered)
        else:
            yield from self.lift_press(arm)
