"""Score measured loading through the lip, followed by a continuously retained load."""
import numpy as np
from scipy.spatial.transform import Rotation

PAN_HANDLE_Y = .245
FLOOR = .7475
NAMES = tuple(f"crumb{i}" for i in range(8))
CORNERS = np.array([(x,y,0.) for x in (-.095,.095) for y in (0.,.15)])

def pose(root):
    return np.asarray(root[:3]), Rotation.from_quat(root[3:]).as_matrix()

def near_grip(hand, grip):
    separation = -sum(hand["fingers"])
    return .006 < separation < .045 and np.linalg.norm(np.asarray(hand["tcp"])-grip) < .035

PIECE_HALF = np.array([[.005+.0005*(i%2),.004,.003] for i in range(8)])
BRISTLE_HALF = np.array([.0463,.005,.018])

def brush_contacts(relative, extents):
    # Candidate contact only. Both the brush and the crumb must also advance.
    center = relative - np.array([0.,0.,.018])
    return np.all(np.abs(center) <= BRISTLE_HALF + extents + .0025, axis=1)

def pushed_pieces(relative, extents, advancing):
    pushed = brush_contacts(relative,extents) & advancing
    # A touching piece may transmit the push to the next one in the pile.
    while pushed.any():
        touching = np.all(np.abs(relative[:,None]-relative[None,:])
                          <= extents[:,None]+extents[None,:]+.0015, axis=2)
        expanded = pushed | (advancing & touching[:,pushed].any(axis=1))
        if np.array_equal(expanded,pushed):
            break
        pushed = expanded
    return pushed

class Scorer:
    def __init__(self):
        self.prev = None
        self.previous_step = 0
        self.loaded = set()
        self.assisted = set()
        self.counts = np.zeros(8, dtype=int)

    def update(self, *, step, objects, root_poses, robot):
        if step != self.previous_step+1:
            raise ValueError("Expected consecutive measured frames")
        self.previous_step = step
        pan, rot = pose(root_poses["pan"])
        brush, brush_rot = pose(root_poses["broom"])
        grips = [pan+rot@[0,PAN_HANDLE_Y,.107], brush+brush_rot@[0,0,.107]]
        hands = list(robot.values())
        two_tools = any(near_grip(hands[i],grips[0]) and near_grip(hands[1-i],grips[1]) for i in (0,1))
        pan_held = any(near_grip(hand,grips[0]) for hand in hands)
        corners = (CORNERS@rot.T+pan)[:,2]-FLOOR
        lip_flush = np.max(np.abs(corners[[0,2]])) < .003
        positions = np.array([root_poses[n][:3] for n in NAMES])
        local = (positions-pan)@rot
        br = (positions-brush)@brush_rot
        piece_rot = np.array([pose(root_poses[n])[1] for n in NAMES])
        pan_half = (np.abs(rot.T @ piece_rot) @ PIECE_HALF[...,None])[...,0]
        brush_half = (np.abs(brush_rot.T @ piece_rot) @ PIECE_HALF[...,None])[...,0]
        for i,(x,y,z) in enumerate(local):
            if any(np.linalg.norm(np.asarray(h["tcp"])-positions[i]) < .032 for h in hands):
                self.assisted.add(i)
            # On leaving the pan a piece must enter through the lip again.
            if abs(x)>.095 or y<0 or y>.153 or z<0 or z-pan_half[i,2]>.042:
                self.loaded.discard(i)
        if self.prev is not None and two_tools and lip_flush:
            previous_local, previous_positions, previous_brush = self.prev
            crossing = ((previous_local[:,1]<=0) & (local[:,1]>0)
                        & (np.abs(local[:,0])<.090) & (local[:,2]>=0) & (local[:,2]<.013))
            if crossing.any() and (brush-previous_brush) @ rot[:,1] > .0001:
                advancing = (positions-previous_positions) @ rot[:,1] > .0001
                pushed = pushed_pieces(br,brush_half,advancing)
                self.loaded.update(np.flatnonzero(crossing & pushed).tolist())
        # CPU PhysX can expose a live tensor view for the brush root. Keep the
        # measured previous position when the simulator updates that buffer.
        self.prev = (local,positions,brush.copy())
        inside = []
        for i,(dx,dy,dz) in enumerate(pan_half):
            x,y,z = local[i]
            inside.append(rot[2,2]>0 and i in self.loaded and i not in self.assisted and abs(x)+dx<=.091
                          and y-dy>=.035 and y+dy<=.147 and .003<=z-dz<=.009)
        lifted = min(corners) >= .05 and pan_held
        self.counts = np.where(np.array(inside) & lifted & ~brush_contacts(br,brush_half),self.counts+1,0)
        # Sixty elapsed 20 Hz intervals require 61 endpoint observations.
        held = int(np.sum(self.counts>=61))
        score = (sum(inside) + held)/16
        return dict(score=float(score),success=held==8,entered=len(self.loaded),
                    inside=int(sum(inside)),held=held,pan_held=bool(pan_held),both_tools_held=bool(two_tools),
                    minimum_clearance_mm=round(float(min(corners))*1000,2))
