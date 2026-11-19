"""Geometry and parts for closing a 1-gallon paint can with a rubber mallet (real size).

Everything is real size; the bench is only the working height. A canvas drop cloth lies on the bench.
On it stands a 1-gallon steel paint can, about half full of tinted latex, its paper label printed for a
made-up maker. Its lid has been set on the ring but not tapped down: it sits on the plug ring's
shoulder about 4 mm proud of seated and a little askew, wet paint in the groove and on the chime. An
8 oz rubber mallet lies on the cloth beside it. The lid is closed by tapping its rim all the way round
with the mallet, while the other hand steadies the can (a half-full gallon on a cloth slides when struck
off-centre).
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  paint can  1 US gallon (3.78 L) open-top steel paint can, ring-and-plug ("triple tight") lid:
             168 mm outside diameter at the chimes (6-5/8 in), 190 mm tall (7-1/2 in), wire bail
             handle with a plastic grip, about 0.30 kg empty, here half full: about 2.4 kg.
  lid        tinplate plug lid, 157 mm across its curl, plug ring 8 mm deep, 0.07 kg.
  mallet     8 oz (227 g) black rubber mallet, head 51 mm across x 92 mm long, wooden handle 25 mm
             thick, 305 mm overall, 0.31 kg.
  drop cloth folded cotton canvas drop cloth, 620 x 500 mm here, with old paint spots.
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

import meshlib as M
from simkit import Part, Piece, Spring

TOP = 0.7475
TEXTURES = Path(__file__).resolve().parent / "textures"

CANVAS = (0.84, 0.80, 0.70)
TIN = (0.80, 0.81, 0.83)
PAINT = (0.23, 0.41, 0.59)              # "Harbor Blue"
PAINT_DARK = (0.17, 0.31, 0.46)
RUBBER = (0.05, 0.05, 0.055)
WOOD = (0.72, 0.55, 0.36)
GRIP = (0.08, 0.08, 0.09)

# -- drop cloth (fixed) --------------------------------------------------------------------------
CLOTH = (0.620, 0.500, 0.004)
CLOTH_C = (0.370, -0.020)
FLOOR = TOP + CLOTH[2]                   # the cloth's top: what the can stands on

# -- paint can (free) ----------------------------------------------------------------------------
CAN_C = (0.340, -0.005)                   # its axis
R_CAN = 0.0842                           # outside radius at the chimes
H_CAN = 0.1905                           # base to the top of the top chime
R_WALL = 0.0836
CAN_MASS = 2.4                           # half full of latex (about 1.8 L at 1.25 kg/L) + the can
TOP_FLAT = 0.182                         # collider: the can's top face (the plug lid bears on it, seated)
LEDGE_TOP = 0.186                        # the plug ring's shoulder: an unseated lid stands this high (its slide's stop)
R_RING_IN = 0.0812                       # inside of the chime ring
R_LID = 0.0800                           # lid collider radius (its curl sits inside the chime)
LID_T = 0.004
PROUD = LEDGE_TOP - TOP_FLAT             # 4 mm: how far the unseated lid stands above seated
LID_MASS = 0.07

# -- mallet (free) -------------------------------------------------------------------------------
HEAD_R = 0.0255
HEAD_L = 0.134
HANDLE_R = 0.0125
HANDLE_L = 0.255                         # head centre to the butt
MALLET_MASS = 0.31
HEAD_C = (0.400, -0.240)                # the head's centre (x, y)
MALLET_YAW = math.radians(-60.0)         # handle heading 120 deg: from the head back toward the arms and left


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def _ring(r_in, r_out, z0, z1, n=24):
    """A ring of box staves (a collider) between radii r_in..r_out and heights z0..z1 (part frame)."""
    out = []
    rm = (r_in + r_out) / 2
    w = 2 * r_out * math.tan(math.pi / n) * 1.02
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(Piece("box", (r_out - r_in, w, z1 - z0), (rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2),
                         rot=(0.0, 0.0, a)))
    return out


# ---------------------------------------------------------------------------------------- cloth
def cloth_vmeshes():
    rnd = np.random.default_rng(3)

    def z(X, Y):
        folds = 0.0016 * np.sin(X * 23.0 + 0.6) * np.cos(Y * 13.0) + 0.0010 * np.sin(Y * 37.0 + X * 7.0)
        ridge = 0.0045 * np.exp(-((Y + 0.17) / 0.014) ** 2) + 0.0030 * np.exp(-((X - 0.20) / 0.010) ** 2)
        wrinkle = 0.0035 * np.exp(-(((X + 0.18) + 0.4 * (Y - 0.05)) / 0.008) ** 2) * (np.abs(Y) < 0.16)
        return CLOTH[2] / 2 + 0.0006 + folds + ridge + wrinkle

    V = [_vm("canvas", M.heightfield(CLOTH[0], CLOTH[1], z, -CLOTH[2] / 2, nx=80, ny=64), CANVAS)]
    hem = M.rbox((CLOTH[0], 0.012, 0.0016), e=4, axis="Z")
    for s in (-1, 1):
        V.append(_vm(f"hem{s}", hem, (0.78, 0.74, 0.64), offset=(0.0, s * (CLOTH[1] / 2 - 0.006), CLOTH[2] / 2 + 0.002)))
    # the near edge folded back over itself (a doubled strip with a rolled crease)
    flap = M.rbox((0.050, CLOTH[1] - 0.04, 0.0030), e=4, axis="Z")
    V.append(_vm("fold_flap", flap, (0.80, 0.76, 0.66), offset=(-CLOTH[0] / 2 + 0.030, 0.0, CLOTH[2] / 2 + 0.0030)))
    crease = M.revolve([(0.0, -(CLOTH[1] - 0.04) / 2), (0.0035, -(CLOTH[1] - 0.04) / 2 + 0.002),
                        (0.0035, (CLOTH[1] - 0.04) / 2 - 0.002), (0.0, (CLOTH[1] - 0.04) / 2)], n=14, axis="Y")
    V.append(_vm("fold_crease", crease, (0.80, 0.76, 0.66), offset=(-CLOTH[0] / 2 + 0.006, 0.0, CLOTH[2] / 2 + 0.0030)))
    # old paint spots: irregular blobs of many sizes, some with flung tails; blue, white, grey
    cols = [PAINT, (0.95, 0.95, 0.93), (0.55, 0.56, 0.58), PAINT_DARK, PAINT]
    for k in range(34):
        x = rnd.uniform(-0.29, 0.29)
        y = rnd.uniform(-0.23, 0.23)
        if math.hypot(x + CLOTH_C[0] - CAN_C[0], y + CLOTH_C[1] - CAN_C[1]) < 0.095:
            continue
        r = float(rnd.choice([rnd.uniform(0.0012, 0.003), rnd.uniform(0.003, 0.008), rnd.uniform(0.008, 0.016)],
                             p=None))
        n = 22
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rad = r * (1.0 + 0.35 * rnd.standard_normal(n).clip(-1.5, 1.5) * 0.6
                   + 0.25 * np.sin(ang * rnd.integers(2, 5) + rnd.uniform(0, 6)))
        tail_dir = rnd.uniform(0, 2 * math.pi)
        if rnd.uniform() < 0.45:                                      # a flung tail
            rad = rad + r * 1.8 * np.exp(-((((ang - tail_dir + math.pi) % (2 * math.pi)) - math.pi) / 0.25) ** 2)
        rim = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.zeros(n)], axis=1)
        pts = np.vstack([[0.0, 0.0, 0.0], rim]) + np.array([x, y, CLOTH[2] / 2 + 0.0034])
        tris = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
        V.append((f"spot{k}", pts, tris, cols[k % len(cols)], None, 1.0))
        if rnd.uniform() < 0.5:                                       # satellite droplets
            for j in range(int(rnd.integers(1, 4))):
                dd = r * rnd.uniform(1.8, 3.5)
                dr = r * rnd.uniform(0.12, 0.3)
                cx = x + dd * math.cos(tail_dir + rnd.uniform(-0.4, 0.4))
                cy = y + dd * math.sin(tail_dir + rnd.uniform(-0.4, 0.4))
                p, t = M.revolve([(0.0, 0.0), (dr, 0.0)], n=10, axis="Z")
                V.append((f"spot{k}_{j}", p + np.array([cx, cy, CLOTH[2] / 2 + 0.0034]), t, cols[k % len(cols)], None, 1.0))
    return tuple(V)


def cloth_object() -> Part:
    return Part("cloth", pos=(CLOTH_C[0], CLOTH_C[1], TOP + CLOTH[2] / 2), size=CLOTH, exact_z=True,
                color=CANVAS, friction=(0.55, 0.45), hide_root=True, vmeshes=cloth_vmeshes())


# ------------------------------------------------------------------------------------------ can
def _label_band():
    """The wrap-around paper label: a textured cylinder just outside the wall, 135 mm tall."""
    n = 96
    r = R_WALL + 0.0004
    z0, z1 = 0.030 - H_CAN / 2, 0.165 - H_CAN / 2
    pts, uv, tris = [], [], []
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        for z, v in ((z0, 0.0), (z1, 1.0)):
            pts.append((r * math.cos(a), r * math.sin(a), z))
            uv.append((1.0 - k / n, v))
    for k in range(n):
        a, b, c, d = 2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1
        tris += [(a, b, c), (a, c, d)]
    # the label's front panel faces the arms (-x): rotate the seam to the back
    pts = np.asarray(pts, float)
    R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[math.cos(-0.45), -math.sin(-0.45), 0], [math.sin(-0.45), math.cos(-0.45), 0], [0, 0, 1]])
    pts = pts @ R.T
    return ("label", pts, np.asarray(tris, int), (1.0, 1.0, 1.0), "tex:" + str(TEXTURES / "can_label.png"), 1.0,
            np.asarray(uv, float))


def can_vmeshes():
    h0 = -H_CAN / 2                      # the part's origin is the can's middle
    body = [(0.0790, 0.0), (0.0832, 0.0006), (0.0842, 0.0030), (0.0846, 0.0060), (0.0838, 0.0090),
            (R_WALL, 0.0110), (R_WALL, 0.0225), (0.0843, 0.0245), (R_WALL, 0.0265),
            (R_WALL, 0.1690), (0.0843, 0.1710), (R_WALL, 0.1730), (R_WALL, 0.1800),
            (0.0844, 0.1830), (0.0850, 0.1865), (0.0846, 0.1898), (0.0836, 0.1905), (0.0826, 0.1895)]
    V = [_vm("wall", M.revolve([(r, h + h0) for r, h in body], n=96), TIN, "steel")]
    V.append(_vm("bottom", M.revolve([(0.0, 0.004 + h0), (0.0790, 0.004 + h0), (0.0800, 0.0008 + h0)], n=64),
                 (0.62, 0.63, 0.65), "steel"))
    # the ring and its groove (the plug lid sits in it), wet with paint
    ring = [(0.0826, 0.1895), (0.0816, 0.1870), (0.0810, 0.1800), (0.0800, 0.1790), (0.0790, 0.1800),
            (0.0786, 0.1860), (0.0780, 0.1870)]
    V.append(_vm("ring", M.revolve([(r, h + h0) for r, h in ring], n=96), (0.86, 0.87, 0.89), "aluminium_brushed"))
    V.append(_vm("rolled_rim", M.revolve([(0.0828, 0.1893 + h0), (0.0840, 0.1909 + h0), (0.0852, 0.1897 + h0), (0.0853, 0.1880 + h0), (0.0846, 0.1868 + h0)], n=96), (0.88, 0.89, 0.91), "aluminium_brushed"))
    groove = [(0.0814, 0.1855), (0.0810, 0.1803), (0.0800, 0.1794), (0.0790, 0.1803), (0.0787, 0.1855)]
    V.append(_vm("groove_paint", M.revolve([(r, h + h0 + 0.0003) for r, h in groove], n=96), PAINT))
    # paint smeared over the chime on one side and running down the wall
    smear = M.revolve([(0.0828, 0.1897 + h0), (0.0838, 0.1907 + h0), (0.0849, 0.1890 + h0), (0.0852, 0.1860 + h0)],
                      n=40, a0=-0.9, a1=0.7)
    V.append(_vm("chime_paint", smear, PAINT))
    rnd = np.random.default_rng(11)
    for k, a in enumerate((-0.75, -0.42, -0.05, 0.31, 0.55, 2.4)):
        L = float(rnd.uniform(0.012, 0.055))
        w = float(rnd.uniform(0.0020, 0.0035))
        r0 = R_WALL + 0.0009
        ca, sa = math.cos(a), math.sin(a)
        drip = M.tube([(r0 * ca, r0 * sa, 0.1850 + h0 - s * L) for s in np.linspace(0.0, 1.0, 8)], w / 2, n=8)
        p, t = drip
        p = p * 1.0
        V.append((f"drip{k}", p, t, PAINT, None, 1.0))
        bulb = M.revolve([(0.0, 0.0), (w * 0.9, w * 0.4), (w * 0.9, w * 1.1), (0.0, w * 1.6)], n=10, axis="Z")
        pb, tb = bulb
        V.append((f"drop{k}", pb + np.array([r0 * ca, r0 * sa, 0.1850 + h0 - L - w]), tb, PAINT, None, 1.0))
    V.append(_label_band())
    # bail ears (stamped tabs, spot-welded) and the wire bail lying down on the +x side, plastic grip
    for s in (1, -1):
        ear = M.rbox((0.014, 0.003, 0.020), e=5, axis="Y")
        V.append(_vm(f"ear{s}", ear, (0.70, 0.71, 0.73), "steel", offset=(0.0, s * (R_WALL + 0.0016), 0.165 + h0)))
    rb = R_WALL + 0.0065
    path = []
    for k in range(41):
        phi = math.pi * k / 40
        path.append((rb * math.sin(phi), rb * math.cos(phi), 0.165 + h0 - 0.052 * math.sin(phi) ** 1.3))
    V.append(_vm("bail", M.tube(path, 0.0016, n=8), (0.66, 0.67, 0.70), "steel"))
    grip = M.revolve([(0.0, -0.045), (0.0072, -0.045), (0.0085, -0.040), (0.0085, 0.040), (0.0072, 0.045),
                      (0.0, 0.045)], n=20, axis="Y")
    V.append(_vm("bail_grip", grip, GRIP, offset=(rb + 0.0012, 0.0, 0.165 + h0 - 0.052)))
    return tuple(V)


def can_object() -> Part:
    h0 = -H_CAN / 2
    P = [Piece("cylinder", (R_CAN, TOP_FLAT, R_CAN), (0.0, 0.0, TOP_FLAT / 2 + h0))]
    P += _ring(R_RING_IN, R_CAN, TOP_FLAT + h0, H_CAN + h0)            # the chime ring round the lid
    return Part("can", pos=(CAN_C[0], CAN_C[1], FLOOR + H_CAN / 2), size=(R_CAN, 0.004, R_CAN), shape="cylinder",
                exact_z=True, mass=CAN_MASS, color=TIN, friction=(0.45, 0.38), restitution=0.05,
                com=(0.0, 0.0, -0.040), contact_offset=0.002, pieces=_colliders(P), hide_root=True,
                vmeshes=can_vmeshes())


# ------------------------------------------------------------------------------------------ lid
def lid_vmeshes():
    """Plug lid, top up, in its own frame (origin at its middle): curl, flat panel, plug ring under it."""
    prof = [(0.0, 0.0010), (0.0640, 0.0010), (0.0680, 0.0004), (0.0720, -0.0012), (0.0760, -0.0010),
            (0.0786, 0.0008), (0.0800, 0.0020), (0.0808, 0.0010), (0.0806, -0.0030), (0.0796, -0.0040)]
    under = [(0.0796, -0.0040), (0.0790, -0.0060), (0.0782, -0.0100), (0.0772, -0.0102), (0.0766, -0.0060),
             (0.0760, -0.0016), (0.0, -0.0016)]
    V = [_vm("lid_top", M.revolve(prof, n=96), (0.80, 0.81, 0.83)),
         _vm("lid_plug", M.revolve(under, n=96), (0.70, 0.71, 0.73))]
    V.append(_vm("lid_channel_paint", M.revolve([(0.0700, -0.0003), (0.0725, -0.0010), (0.0752, -0.0008), (0.0775, -0.0001)], n=96), PAINT))
    # paint wiped on the lid's rim and curl (it was the lid of a used can)
    V.append(_vm("lid_paint", M.revolve([(0.0742, -0.0008), (0.0770, -0.0006), (0.0794, 0.0012), (0.0803, 0.0021)],
                                        n=48, a0=0.3, a1=2.6), PAINT))
    V.append(_vm("lid_paint2", M.revolve([(0.0760, -0.0007), (0.0790, 0.0010), (0.0803, 0.0022)], n=24, a0=3.6, a1=4.4),
                 PAINT))
    # tint-recipe sticker on the panel
    w, h = 0.060, 0.040
    pts = np.array([[-w / 2, -h / 2, 0.0013], [w / 2, -h / 2, 0.0013], [w / 2, h / 2, 0.0013], [-w / 2, h / 2, 0.0013]])
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])        # reads from the arms' side
    pts = pts @ rot.T + np.array([0.012, 0.0, 0.0])
    V.append(("sticker", pts, np.array([[0, 1, 2], [0, 2, 3]]), (1.0, 1.0, 1.0),
              "tex:" + str(TEXTURES / "lid_sticker.png"), 1.0, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)))
    return tuple(V)


def lid_pos():
    """The unseated lid's middle: resting on the plug ring's shoulder (4 mm proud), centred on the can."""
    return (CAN_C[0], CAN_C[1], FLOOR + LEDGE_TOP + LID_T / 2)


def lid_seated_z():
    return FLOOR + TOP_FLAT + LID_T / 2


def lid_top_unseated():
    return FLOOR + LEDGE_TOP + LID_T


CAN_POS = (CAN_C[0], CAN_C[1], FLOOR + H_CAN / 2)
DROP = -(PROUD + 0.0008)                 # the slide's lower stop once every sector is seated (rests on the can)


def lid_object() -> Part:
    """The plug lid, its plug in the ring: a vertical slide on the can (the ring's groove holds it
    sideways), its lower stop 4 mm proud of seated until it is tapped down all round (task.Dynamics)."""
    return Part("lid", pos=lid_pos(), size=(R_LID, LID_T, R_LID), shape="cylinder", exact_z=True, mass=LID_MASS,
                color=TIN, friction=(0.40, 0.32), restitution=0.0, contact_offset=0.0015, hide_root=True,
                vmeshes=lid_vmeshes(),
                spring=Spring(axis="Z", lower=0.0, upper=0.003, stiffness=0.0, damping=2.0, target=0.0,
                              parent="can", parent_pos=CAN_POS))


# --------------------------------------------------------------------------------------- mallet
def mallet_dir():
    """Unit vector from the head's centre toward the handle's butt (in the bench plane)."""
    return np.array([-math.cos(MALLET_YAW), -math.sin(MALLET_YAW), 0.0])


def _mallet_rot():
    """Part frame: x along the handle (head -> butt), z up; the head's axis is the part's y."""
    d = mallet_dir()
    y = np.cross([0.0, 0.0, 1.0], d)
    return np.stack([d, y, [0.0, 0.0, 1.0]], axis=1)


def mallet_vmeshes():
    """In the mallet's frame: origin at the head's centre, handle along +x, head axis along y."""
    head = M.revolve([(0.0, -HEAD_L / 2), (HEAD_R - 0.004, -HEAD_L / 2), (HEAD_R, -HEAD_L / 2 + 0.004),
                      (HEAD_R, HEAD_L / 2 - 0.004), (HEAD_R - 0.004, HEAD_L / 2), (0.0, HEAD_L / 2)], n=48, axis="Y")
    V = [_vm("head", head, RUBBER, "abs")]
    for s in (-1, 1):                                                    # moulded parting lines on the faces
        ring = M.revolve([(HEAD_R - 0.010, 0.0), (HEAD_R - 0.009, 0.0)], n=32, axis="Y")
        V.append(_vm(f"face{s}", ring, (0.15, 0.15, 0.16), offset=(0.0, s * (HEAD_L / 2 + 0.0002), 0.0)))
    prof = [(0.0, 0.010), (0.0105, 0.010), (0.0110, 0.060), (0.0118, 0.150), (0.0128, 0.215), (0.0132, 0.240),
            (0.0125, 0.251), (0.0090, 0.2555), (0.0, 0.2560)]
    V.append(_vm("handle", M.revolve(prof, n=24, axis="Z"), WOOD, "oak",
                 R=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])))
    return tuple(V)


def mallet_object() -> Part:
    R = _mallet_rot()
    # pieces are axis-aligned in the part frame only via ``rot``; the part's own yaw sets its heading
    P = [Piece("cylinder", (HANDLE_R, HANDLE_L - 0.012, HANDLE_R), ((HANDLE_L + 0.012) / 2, 0.0, 0.0), axis="X")]
    yaw = math.atan2(R[1, 0], R[0, 0])
    z = FLOOR + HEAD_R
    return Part("mallet", pos=(HEAD_C[0], HEAD_C[1], z), size=(HEAD_R, HEAD_L, HEAD_R), shape="cylinder", axis="Y",
                exact_z=True, mass=MALLET_MASS, yaw=yaw, color=RUBBER, friction=(0.9, 0.8),
                restitution=0.15, com=(0.035, 0.0, 0.0), contact_offset=0.002, pieces=_colliders(P), hide_root=True,
                vmeshes=mallet_vmeshes())


# --------------------------------------------------------------------------------- measurements
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
