"""``simkit`` -- the extra scene machinery this task needs, installed by patching at import time.

A ``yam_arena`` task file may only list ``Box``, upright ``Cylinder``, ``Sphere`` and library ``Asset``
objects, each one rigid body with one primitive collider, one flat sRGB colour and the one shared
friction material. This task needs more than that, so this module adds it **without touching
``yam_arena`` or ``isaaclab_arena``**: both stay at their pinned versions, and everything here ships
inside the task folder and installs itself when the task file is imported.

What it adds
------------
``Part``   a ``Box`` subclass (so ``task_file.load``'s ``isinstance`` check still passes) that also
           carries an MDL material, an opacity, its own friction/restitution, a collision
           approximation, a shape other than a box, and child ``Piece``s welded into one rigid body.
``Piece``  one more bit of geometry under the same rigid body: this is how a hollow or concave tool
           is built, since PhysX wants convex colliders.

How the patches work
--------------------
``use_production_assets()``
    Arena 0.3.x rewrites its object-library root to the ``omniverse-content-staging`` bucket, which no
    longer serves those files -- its own ``assets/nucleus.py`` still carries
    ``TODO(2026.07.14, Point Arena assets to the production bucket before release)``. Every library
    asset, and the stock ``station`` scene with it, therefore fails to load with
    ``Failed to open layer @https://omniverse-content-staging...``. The identical paths are served
    from ``omniverse-content-production``, so the host is rewritten when an asset is looked up.

``patch_scene()``
    ``yam_arena.scene.task_object`` turns one task-file spec into one Arena object. The task file is
    imported before Isaac Sim starts, so ``yam_arena.scene`` (which imports ``isaaclab.sim``) cannot
    be imported yet; instead a ``sys.meta_path`` finder wraps that module's loader and applies the
    patch the moment the embodiment imports it, after Kit is up. The patched ``task_object`` builds
    ``Part`` specs itself and hands everything else to the original.

``patch_demo()``
    ``demo.Waypoint.yaw`` is a float that ``demo.grip_rotation`` turns into a straight-down grip
    rotated about the vertical, which cannot tilt the wrist. ``ArmPlan`` already solves IK for a full
    TCP rotation, so ``grip_rotation`` is wrapped to pass a 3x3 matrix straight through while a float
    still reaches the original. A waypoint may then carry ``rot_y(-0.6) @ DOWN`` to tilt the gripper.
    (NumPy only, so this patch applies immediately.)

Nothing here mutates a pinned module's source; each patch is idempotent and re-entrant-safe.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field, replace

import numpy as np

from yam_arena.task_file import Asset, Box

# -- MDL materials -----------------------------------------------------------------------
# Isaac Sim's material library, on the production bucket (checked reachable). Visual only: the
# physics material is separate, and set per Part.
_NVIDIA = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/NVIDIA"
MDL = {
    "aluminium": f"{_NVIDIA}/Materials/Base/Metals/Aluminum_Anodized.mdl",
    "aluminium_brushed": f"{_NVIDIA}/Materials/vMaterials_2/Metal/Aluminum_Brushed.mdl",
    "steel": f"{_NVIDIA}/Materials/Base/Metals/Steel_Stainless.mdl",
    "steel_painted": f"{_NVIDIA}/Materials/Base/Metals/Steel_Carbon.mdl",
    "oak": f"{_NVIDIA}/Materials/Base/Wood/Oak.mdl",
    "abs": f"{_NVIDIA}/Materials/Base/Plastics/Plastic_ABS.mdl",
    "acrylic": f"{_NVIDIA}/Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl",
    "glass": f"{_NVIDIA}/Materials/Base/Glass/Clear_Glass.mdl",
}

# Realistic (static, dynamic) friction and restitution per real material pairing.
FRICTION = {
    "steel_on_steel": (0.60, 0.50),
    "aluminium_on_steel": (0.55, 0.45),
    "abs_on_steel": (0.40, 0.33),
    "wood_on_wood": (0.45, 0.35),
    "wood_on_steel": (0.40, 0.32),
    "rubber": (1.20, 1.00),  # yam_arena's shared default, for anything meant to be gripped
}
RESTITUTION = {"steel": 0.45, "aluminium": 0.40, "abs": 0.25, "wood": 0.30, "rubber": 0.10}

SHAPES = ("box", "cylinder", "cone", "sphere", "capsule")
APPROX = ("convexHull", "convexDecomposition", "sdf", "triangleMesh", "none")


# -- specs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One more bit of geometry welded into the owning ``Part``'s rigid body.

    ``offset`` is its centre relative to the ``Part``'s ``pos``. ``size`` reads as the shape needs:
    a box takes its (x, y, z) edges; a cylinder, cone or capsule takes ``(radius, height, radius)``;
    a sphere takes ``(radius, radius, radius)``. Every component must be positive, because
    ``task_file.Box`` validates all three -- hence the repeated radius rather than a placeholder.
    ``axis`` is the cylinder/cone/capsule axis ("X", "Y" or "Z").

    ``rot`` turns the piece about its own centre, as (roll, pitch, yaw) in radians applied in that
    order. Without it a ``Part`` can only ever be a stack of axis-aligned blocks, which is enough
    for a tray or a rack but not for anything with a slope -- the sloping wall of a funnel, say, or
    the staves of a round vessel.
    """

    shape: str = "box"
    size: tuple[float, float, float] = (0.01, 0.01, 0.01)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "Z"
    color: tuple[float, float, float] | None = None  # None: the Part's colour
    material: str | None = None  # None: the Part's material
    opacity: float | None = None
    emissive: tuple[float, float, float] | None = None
    collide: bool = True  # False: visual trim only
    hidden: bool = False  # True: a collider only, not rendered (inside a detailed visual mesh)
    # None: share the Part's physics material. A vessel wants two: the outside has to be grippy
    # enough for the jaws, the inside slippery enough that its contents run out when it is tipped.
    friction: tuple[float, float] | None = None
    restitution: float | None = None

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"Piece.shape must be one of {SHAPES}, got {self.shape!r}")
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"Piece.axis must be X, Y or Z, got {self.axis!r}")


@dataclass(frozen=True)
class Lifted(Asset):
    """A library ``Asset`` raised off the bench by ``lift`` metres.

    ``Asset`` always settles a model onto the table -- ``station.placement`` solves its z from the
    model's own bounds -- so a tool meant to sit on a rack ends up under it instead. This carries the
    extra height, which the patched ``task_object`` adds to the pose the original built.
    """

    lift: float = 0.0


@dataclass(frozen=True)
class Anchor:
    """A world-anchored fixed joint that lets go once it is pulled hard enough.

    For a tool that has to *stand* somewhere before it is picked up. A funnel is a cone on a narrow
    spout and falls over on its own; propping it up with a visible cradle puts a piece of hardware
    in the scene that has nothing to do with the task, and an invisible collider is not actually
    invisible -- a zero-opacity preview surface still renders. A fixed joint holds the tool upright
    with no geometry at all, and ``break_force``/``break_torque`` make it yield to the arm: set them
    well above the tool's own weight and well below what the gripper can pull.
    """

    break_force: float = 6.0  # newtons
    break_torque: float = 1.5  # newton-metres


@dataclass(frozen=True)
class Spring:
    """A world-anchored prismatic joint with a spring drive, so a free ``Part`` returns to rest.

    Task files have no joints, which is why a spring-loaded button normally cannot be modelled. PhysX
    does simulate a single joint on a plain rigid body without an articulation root, so one is defined
    in the spawner: the body may slide along ``axis`` between ``lower`` and ``upper`` metres of its
    start, and a force drive pulls it back to ``target`` with ``stiffness`` (N/m) and ``damping``
    (N s/m). Isaac Lab still sees an ordinary rigid body, so ``object_states`` reports its pose as usual.
    """

    axis: str = "X"
    lower: float = 0.0
    upper: float = 0.020
    stiffness: float = 220.0
    damping: float = 6.0
    target: float = 0.0
    # Sliding on another task object instead of the world (a hose clamp on its hose): that sibling's
    # name and its spawn position (its spawn orientation must be identity).
    parent: str | None = None
    parent_pos: tuple[float, float, float] | None = None
    # World-anchored at ``anchor`` (stage coordinates) on the body's point ``local`` (its own frame),
    # instead of at its spawn position; ``enabled`` False defines the joint switched off, for the task
    # to switch on at run time (``physics:jointEnabled``) -- a hose that grips a barb once pushed on.
    anchor: tuple[float, float, float] | None = None
    local: tuple[float, float, float] | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"Spring.axis must be X, Y or Z, got {self.axis!r}")
        if self.upper < self.lower:
            raise ValueError("Spring.upper must be >= Spring.lower")


@dataclass(frozen=True)
class Hinge:
    """A world-anchored revolute joint about a world axis through ``pivot``, with a damping drive.

    For a mount that turns by hand and stays where it is left (an articulated coolant nozzle on its
    swivel): with no gravity torque about a vertical axis, damping alone holds the angle.
    """

    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "Z"
    damping: float = 0.05      # N m s / rad
    lower_deg: float = -180.0
    upper_deg: float = 180.0
    friction: float = 0.0      # PhysX joint friction: holds the angle where it is left against small loads
    stiffness: float = 0.0     # N m / rad toward target_deg (0: no spring)
    target_deg: float = 0.0
    # Hinged to another task object instead of the world (a flap on a wall panel): that sibling's
    # name and its spawn position (its spawn orientation must be identity).
    parent: str | None = None
    parent_pos: tuple[float, float, float] | None = None
    # Joint frame orientation (w, x, y, z) in both bodies' spawn frames: ``axis`` is taken in this
    # frame, so a hinge can turn about any direction, not only a world axis.
    frame_rot: tuple[float, float, float, float] | None = None
    # A joint made at spawn but switched on later (a nut that only meets its thread once it is put
    # there): its frame in body 1 is given directly (``local1``), each body's frame orientation
    # separately (``rot0``/``rot1``, w x y z), and it starts disabled unless ``enabled``.
    local1: tuple[float, float, float] | None = None
    rot0: tuple[float, float, float, float] | None = None
    rot1: tuple[float, float, float, float] | None = None
    local0: tuple[float, float, float] | None = None
    enabled: bool = True


@dataclass(frozen=True)
class Part(Box):
    """A task-file object with a real material, a real collider and, optionally, several pieces.

    It subclasses ``Box`` so ``yam_arena.task_file.load`` accepts it and validates ``pos``, ``size``,
    ``color``, ``mass`` and ``yaw`` as usual. ``size`` and ``shape`` describe the *root* piece, which
    sits at ``pos`` and carries the rigid body; ``pieces`` are welded to it, so a ``Part`` with a
    ``mass`` is one rigid body however many pieces it has.

    ``size`` follows :class:`Piece`: box edges (x, y, z), or ``(radius, height, radius)`` for a
    cylinder, cone or capsule, or ``(radius, radius, radius)`` for a sphere. All three components
    must be positive, since ``Box`` validates them.
    """

    shape: str = "box"
    axis: str = "Z"
    material: str | None = None  # a key of MDL; None: a plain coloured surface
    opacity: float = 1.0  # < 1 renders see-through (the collider is unaffected)
    emissive: tuple[float, float, float] | None = None  # self-lit, e.g. an illuminated button
    friction: tuple[float, float] | None = None  # None: yam_arena's shared rubber-like default
    restitution: float | None = None
    collision: str = "convexHull"
    hull_count: int = 16  # convexDecomposition: cap on hulls for this body
    hull_voxels: int = 500_000  # convexDecomposition: voxel resolution
    min_thickness: float = 0.0015  # convexDecomposition: keeps thin walls from being lost
    sdf_resolution: int = 256
    # PhysX starts generating contacts this far out.
    contact_offset: float = 0.005
    pieces: tuple[Piece, ...] = field(default=())
    # Arena re-centres a body on the bench: whatever ``pos`` says, the origin lands at the bench
    # top plus half the body's height.
    exact_z: bool = False
    spring: Spring | None = None  # free parts only: makes this body spring-return along one axis
    # Free parts only: holds the body where it starts until something pulls hard enough to break it.
    anchor: Anchor | None = None
    # Where the body balances, in its own frame.
    com: tuple[float, float, float] | None = None
    # Free parts only: cap on the speed PhysX may use to push this body out of an overlap. Light
    # granular bodies squeezed between a tool and a wall otherwise get flung away.
    max_depenetration: float | None = None
    hinge: Hinge | None = None  # free parts only: turn about a world axis, damped
    hinge2: Hinge | None = None  # a second revolute joint (e.g. made at spawn, switched on later)
    # Detailed visual meshes (no collision): (name, points (N, 3) in the Part's frame, triangles (M, 3),
    # colour, material key or None, opacity). The colliders stay simple primitives, hidden inside.
    vmeshes: tuple = field(default=())
    hide_root: bool = False  # the root primitive is a collider only, not rendered

    def _check(self) -> None:
        super()._check()
        if self.shape not in SHAPES:
            raise ValueError(f"{self.name}.shape must be one of {SHAPES}, got {self.shape!r}")
        if self.collision not in APPROX:
            raise ValueError(f"{self.name}.collision must be one of {APPROX}, got {self.collision!r}")
        if self.material is not None and self.material not in MDL:
            raise ValueError(f"{self.name}.material must be one of {sorted(MDL)}, got {self.material!r}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"{self.name}.opacity must be in 0..1, got {self.opacity!r}")
        for p in self.pieces:
            if p.material is not None and p.material not in MDL:
                raise ValueError(f"{self.name}: piece material {p.material!r} is not in MDL")
        if self.spring is not None and not self.free:
            raise ValueError(f"{self.name}: a spring needs a mass, or there is no body to return")
        if self.anchor is not None and not self.free:
            raise ValueError(f"{self.name}: an anchor needs a mass; a fixed part is already fixed")

    def with_pieces(self, *pieces: Piece) -> Part:
        return replace(self, pieces=tuple(self.pieces) + pieces)


# -- tilted grips (NumPy only; applies at import) ----------------------------------------


def rot_y(angle: float) -> np.ndarray:
    """Rotation about world y, for tilting a grip forward (negative) or back (positive)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def patch_demo(smooth: bool = True) -> np.ndarray:
    """Let a ``Waypoint`` carry a 3x3 TCP rotation in its ``yaw`` field, and optionally make ``ArmPlan``
    follow a smooth curve. Returns the straight-down grip."""
    from yam_arena import demo

    if not getattr(demo, "_simkit_patched", False):
        original = demo.grip_rotation

        def grip_rotation(yaw=0.0):
            arr = np.asarray(yaw)
            if arr.ndim == 2 and arr.shape == (3, 3):
                return arr.astype(float)
            return original(float(yaw))

        grip_rotation.__doc__ = original.__doc__
        demo.grip_rotation = grip_rotation
        demo._simkit_original_grip_rotation = original
        demo._simkit_patched = True
    if smooth:
        patch_arm_plan()
    return np.asarray(demo.grip_rotation(0.0), dtype=float)


def _ease_time(t: float, span: float, ramp: float = 0.5) -> float:
    """Elapsed time warped by a trapezoidal speed profile: eased at both ends, true in between.

    A single smoothstep across a whole move does ease the start and stop, but it also rewrites every
    waypoint's timing -- it is slowest at the ends and fastest in the middle, so a waypoint placed
    mid-move to dwell on a pose is rushed through exactly when the plan asked to wait. Ramping the
    speed over half a second at each end instead leaves the clock alone everywhere else.
    """
    span = float(span)
    if span <= 0.0:
        return 0.0
    r = min(float(ramp), span / 2.0)
    if r <= 0.0:
        return float(np.clip(t, 0.0, span))
    peak = span / (span - r)  # so the profile still covers the whole move in the time given
    t = float(np.clip(t, 0.0, span))
    if t < r:
        return peak * t * t / (2.0 * r)
    if t <= span - r:
        return peak * (r / 2.0 + (t - r))
    return span - peak * (span - t) ** 2 / (2.0 * r)


def _locate(durations, s: float) -> tuple[int, float]:
    """(segment index, local 0..1) at normalised progress ``s`` through a move, by *time*.

    Each waypoint's own duration is its share of the move, which is what the plan asks for. The
    obvious alternative -- splitting by arc length -- quietly breaks two things: a waypoint repeated
    to dwell on a pose has zero length and so gets no time at all, and a short, slow, careful move is
    rushed through while a long sweep is given the clock. Both showed up as the scoop being dragged
    across a dish 16 mm above the floor it had been told to rest on.
    """
    total = float(sum(durations))
    if total <= 0.0:
        return max(len(durations) - 1, 0), 1.0
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(durations, dtype=float))]) / total
    s = float(np.clip(s, 0.0, 1.0))
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(durations) - 1))
    span = cum[i + 1] - cum[i]
    return i, 1.0 if span < 1e-12 else (s - cum[i]) / span


def _catmull_rom(points: np.ndarray, i: int, u: float) -> np.ndarray:
    """Catmull-Rom point on segment ``i`` at local ``u``.

    Straight segments between waypoints make a plan turn corners, which looks mechanical and jerks the
    wrist. A Catmull-Rom spline passes through every waypoint but rounds the corners between them, so
    the same waypoints describe a flowing path. The ends are duplicated, so the curve starts and
    finishes exactly on the first and last waypoint.
    """
    p0 = points[i - 1] if i - 1 >= 0 else points[i]
    p1, p2 = points[i], points[i + 1]
    if float(np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float))) < 1e-9:
        return np.asarray(p1, float).copy()          # a repeated waypoint is a dwell: hold it exactly
    p3 = points[i + 2] if i + 2 < len(points) else points[i + 1]
    u2, u3 = u * u, u * u * u
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )


def patch_arm_plan() -> None:
    """Make ``ArmPlan`` glide along a spline through its waypoints, easing once per move.

    Stock ``ArmPlan`` interpolates each waypoint pair as a straight line and applies a smoothstep to
    every segment, so the TCP stops dead at every waypoint and turns square corners. Here consecutive
    waypoints that share a finger target are treated as one *move*: a single smoothstep spans the
    whole move and the TCP follows a Catmull-Rom spline through its waypoints, each waypoint getting
    the share of the move its own duration asks for. The result accelerates once, flows through the
    via points and decelerates once, while still dwelling where the plan says to dwell.

    ``yam_arena.scripted`` binds the ``ArmPlan`` class itself, so patching the method on the class
    reaches the policy's copy; nothing is rebound by name.
    """
    from yam_arena import demo, ik

    if getattr(demo.ArmPlan, "_simkit_smooth", False):
        return

    def moves(self):
        """[(waypoints, start_pos, start_rot, start_finger)] -- the plan split into smooth moves.

        A waypoint that changes the finger target is a gripper action and becomes a move of its own,
        so the jaws still open or close over exactly the time that waypoint was given. Every run of
        waypoints that leaves the finger alone becomes one move, travelled as a single spline.
        """
        cached = getattr(self, "_simkit_moves", None)
        if cached is not None:
            return cached
        groups, run = [], []
        for w in self.waypoints:
            previous = run[-1].finger if run else (groups[-1][-1].finger if groups else self.finger_start)
            if w.finger != previous:
                if run:
                    groups.append(run)
                    run = []
                groups.append([w])
            else:
                run.append(w)
        if run:
            groups.append(run)
        out = []
        pos, rot, finger = self.start_pos, self.start_rot, self.finger_start
        for group in groups:
            out.append((group, pos, rot, finger))
            last = group[-1]
            pos, rot, finger = np.asarray(last.pos, float), demo.grip_rotation(last.yaw), last.finger
        self._simkit_moves = out
        return out

    def target(self, t: float):
        for run, pos0, rot0, finger0 in moves(self):
            span = sum(w.duration for w in run)
            if t < span or run is moves(self)[-1][0]:
                if span <= 0:
                    last = run[-1]
                    return np.asarray(last.pos, float), demo.grip_rotation(last.yaw), last.finger, last.label
                u = _ease_time(t, span) / span
                points = np.vstack([pos0, *[np.asarray(w.pos, float) for w in run]])
                rots = [rot0, *[demo.grip_rotation(w.yaw) for w in run]]
                last = run[-1]
                # Which waypoint's label is current, by elapsed time rather than by distance.
                elapsed, label = min(t, span), run[-1].label
                for w in run:
                    if elapsed < w.duration:
                        label = w.label
                        break
                    elapsed -= w.duration
                i, local = _locate([w.duration for w in run], u)
                return (
                    _catmull_rom(points, i, local),
                    # The orientation follows the same via points as the path, so a move may turn the
                    # wrist through several attitudes -- tipping a scoop, say -- inside one glide.
                    ik.slerp_matrix(rots[i], rots[i + 1], local),
                    finger0 + u * (last.finger - finger0),
                    label if t < span else "done",
                )
            t -= span
        return self.start_pos, self.start_rot, self.finger_start, "done"

    demo.ArmPlan.target = target
    demo.ArmPlan._simkit_smooth = True


# -- asset root --------------------------------------------------------------------------

_STALE, _LIVE = "omniverse-content-staging", "omniverse-content-production"


def use_production_assets() -> bool:
    """Serve Arena's object library from the production bucket. True if this call installed the fix."""
    registries = importlib.import_module("isaaclab_arena.assets.registries")
    registry = registries.AssetRegistry
    if getattr(registry, "_simkit_production_assets", False):
        return False
    original = registry.get_asset_by_name

    def get_asset_by_name(self, name, *args, **kwargs):
        asset = original(self, name, *args, **kwargs)
        path = getattr(asset, "usd_path", None)
        if isinstance(path, str) and _STALE in path:
            try:
                asset.usd_path = path.replace(_STALE, _LIVE)
            except Exception:  # a read-only asset spec: leave it to fail loudly at spawn
                pass
        return asset

    registry.get_asset_by_name = get_asset_by_name
    registry._simkit_production_assets = True
    return True


# -- patching a module that is imported after Isaac Sim starts ---------------------------


def _patch_on_import(module_name: str, apply) -> None:
    """Call ``apply(module)`` as soon as ``module_name`` is imported -- or right now if it already is."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        apply(existing)
        return

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            apply(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module_name:
                return None
            # Ask every other finder, so this one never recurses into itself.
            for finder in [f for f in sys.meta_path if f is not self]:
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _Loader(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())


def patch_scene() -> None:
    """Teach ``yam_arena.scene.task_object`` to build ``Part`` specs, whenever that module loads."""
    _patch_on_import("yam_arena.scene", _install_task_object)


def _install_task_object(scene) -> None:
    if getattr(scene, "_simkit_patched", False):
        return
    # Arena looks assets up while it builds the scene, which is after Kit has started -- so this is
    # the right moment to repair the asset root, and it keeps the task file free of Isaac imports.
    use_production_assets()
    original = scene.task_object

    # A Lifted asset's height has to be injected *before* the Object is constructed.
    from yam_arena import station

    real_placement = station.placement
    pending: dict = {"lift": 0.0, "z": None}

    def placement(item, lo, hi):
        scale, pos, rot = real_placement(item, lo, hi)
        if pending["z"] is not None:
            pos = (pos[0], pos[1], pending["z"])
        elif pending["lift"]:
            pos = (pos[0], pos[1], pos[2] + pending["lift"])
        return scale, pos, rot

    placement.__doc__ = real_placement.__doc__
    station.placement = placement

    def task_object(spec):
        if isinstance(spec, Part):
            pending["z"] = float(spec.pos[2]) if spec.exact_z else None
            try:
                return _build_part(spec)
            finally:
                pending["z"] = None
        pending["lift"] = float(getattr(spec, "lift", 0.0) or 0.0)
        try:
            return original(spec)
        finally:
            pending["lift"] = 0.0

    task_object.__doc__ = original.__doc__
    scene.task_object = task_object
    scene._simkit_original_task_object = original
    scene._simkit_patched = True


# -- building a Part ---------------------------------------------------------------------


def _mesh_cfg(sim_utils, shape: str, size, axis: str, **body):
    """The spawner cfg for one piece. Mesh shapes are used throughout so that every collider can take
    a mesh approximation (a convex decomposition or an SDF), which the analytic shapes cannot."""
    if shape == "box":
        return sim_utils.MeshCuboidCfg(size=tuple(size), **body)
    if shape == "cylinder":
        return sim_utils.MeshCylinderCfg(radius=size[0], height=size[1], axis=axis, **body)
    if shape == "cone":
        return sim_utils.MeshConeCfg(radius=size[0], height=size[1], axis=axis, **body)
    if shape == "capsule":
        return sim_utils.MeshCapsuleCfg(radius=size[0], height=size[1], axis=axis, **body)
    return sim_utils.MeshSphereCfg(radius=size[0], **body)


def _euler_xyzw(rot) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a (roll, pitch, yaw) triple -- the order the mesh spawners want.

    Scalar last, not first. Isaac Lab's own ``Pose`` and these spawners both document xyzw, and
    passing wxyz is not rejected -- it is silently read as xyzw, which turns a small turn about the
    vertical into a large one about x. Every stave of a round vessel then lies over on its side,
    and a ring of them comes out as a starburst.
    """
    roll, pitch, yaw = (float(v) for v in rot)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _raw_spawner(shape: str):
    """The undecorated spawner for a shape.

    ``cfg.func`` reaches the function through the config instance, which binds it as a method, and
    Isaac Lab's ``@clone`` wrapper replicates across envs and would re-clone a child prim. Taking the
    module-level function and unwrapping ``@clone`` spawns exactly one prim where asked -- the same
    thing ``yam_arena.scene`` does with ``spawn_from_usd.__wrapped__``.
    """
    from isaaclab.sim.spawners.meshes import meshes

    return {
        "box": meshes.spawn_mesh_cuboid,
        "cylinder": meshes.spawn_mesh_cylinder,
        "cone": meshes.spawn_mesh_cone,
        "capsule": meshes.spawn_mesh_capsule,
        "sphere": meshes.spawn_mesh_sphere,
    }[shape].__wrapped__


def _look(sim_utils, color, material: str | None, opacity: float, emissive=None):
    """A visual material: an MDL from Isaac's library, or a plain surface.

    Opacity and emission are only available on the preview surface, so a part that asks for either
    gets one even if it also names an MDL.
    """
    from yam_arena.station import linear_rgb

    if material in ("matte", "satin", "gloss"):                          # plain surfaces of set roughness
        return sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(tuple(color)), opacity=float(opacity),
                                           roughness={"matte": 0.92, "satin": 0.6, "gloss": 0.18}[material],
                                           metallic=0.0)
    if material is not None and opacity >= 1.0 and emissive is None:
        return sim_utils.MdlFileCfg(mdl_path=MDL[material], project_uvw=True, albedo_brightness=1.0)
    look = sim_utils.PreviewSurfaceCfg(
        diffuse_color=linear_rgb(tuple(color)),
        roughness=0.35 if material in ("steel", "aluminium", "glass", "acrylic") else 0.5,
        metallic=0.8 if material in ("steel", "aluminium", "aluminium_brushed", "steel_painted") else 0.0,
        opacity=float(opacity),
    )
    if emissive is not None:
        look.emissive_color = linear_rgb(tuple(emissive))
    return look


def _collision_props(spec: Part):
    """Collision properties carrying the chosen mesh approximation.

    Contacts start 5 mm out and rest at zero separation, matching ``yam_arena.scene``, so thin finger
    pads register contact before they penetrate.
    """
    from isaaclab_physx.sim.schemas import (
        MeshCollisionPropertiesCfg,
        PhysxCollisionPropertiesCfg,
        PhysxConvexDecompositionPropertiesCfg,
        PhysxSDFMeshPropertiesCfg,
    )

    if spec.collision == "none":
        return None
    if spec.collision == "convexDecomposition":
        mesh = PhysxConvexDecompositionPropertiesCfg(
            mesh_approximation_name="convexDecomposition",
            hull_vertex_limit=64,
            max_convex_hulls=int(spec.hull_count),
            min_thickness=float(spec.min_thickness),
            voxel_resolution=int(spec.hull_voxels),
            error_percentage=1.0,
            shrink_wrap=True,
        )
    elif spec.collision == "sdf":
        mesh = PhysxSDFMeshPropertiesCfg(
            mesh_approximation_name="sdf",
            sdf_margin=0.0,
            sdf_narrow_band_thickness=0.01,
            sdf_resolution=int(spec.sdf_resolution),
            sdf_subgrid_resolution=6,
        )
    else:
        mesh = MeshCollisionPropertiesCfg(mesh_approximation_name=spec.collision)
    return PhysxCollisionPropertiesCfg(
        contact_offset=float(spec.contact_offset), rest_offset=0.0, mesh_collision_property=mesh
    )


def _physics_material(sim_utils, spec, fallback=None):
    """The rigid-body material for a ``Part`` or one of its ``Piece``s.

    A piece falls back to its owning Part's material, and the Part to Arena's grippy default.
    """
    from yam_arena.constants import GRIPPY_FRICTION

    friction = spec.friction
    if friction is None:
        friction = fallback.friction if fallback is not None else None
    static, dynamic = friction if friction is not None else GRIPPY_FRICTION
    kwargs = dict(static_friction=float(static), dynamic_friction=float(dynamic))
    restitution = spec.restitution
    if restitution is None and fallback is not None:
        restitution = fallback.restitution
    if restitution is not None:
        kwargs["restitution"] = float(restitution)
    return sim_utils.RigidBodyMaterialCfg(**kwargs)


def _add_spring(prim, spec: Part) -> None:
    """Anchor the body to the world with a prismatic joint whose drive springs it back to rest."""
    from pxr import Gf, Sdf, UsdPhysics

    spring = spec.spring
    path = f"{prim.GetPath()}/spring_joint"
    joint = UsdPhysics.PrismaticJoint.Define(prim.GetStage(), path)
    joint.CreateAxisAttr(spring.axis)
    # body0 is left unset, which anchors the joint to the world; its frame is then given in stage
    # coordinates, and this scene is built with one environment whose origin is the world origin.
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(prim.GetPath()))])
    if spring.parent is not None:
        joint.CreateBody0Rel().SetTargets([prim.GetPath().GetParentPath().AppendChild(spring.parent)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(a) - float(b) for a, b in zip(spec.pos, spring.parent_pos))))
    elif spring.anchor is not None:
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in spring.anchor)))
    else:
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in spec.pos)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in (spring.local or (0.0, 0.0, 0.0)))))
    if not spring.enabled:
        joint.CreateJointEnabledAttr(False)
    joint.CreateLowerLimitAttr(float(spring.lower))
    joint.CreateUpperLimitAttr(float(spring.upper))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(spring.stiffness))
    drive.CreateDampingAttr(float(spring.damping))
    drive.CreateTargetPositionAttr(float(spring.target))


def _add_hinge(prim, spec: Part, h=None, name="hinge_joint") -> None:
    """Anchor the body to the world with a revolute joint and an angular damping drive."""
    from pxr import Gf, Sdf, UsdPhysics

    h = spec.hinge if h is None else h
    joint = UsdPhysics.RevoluteJoint.Define(prim.GetStage(), f"{prim.GetPath()}/{name}")
    joint.CreateAxisAttr(h.axis)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(prim.GetPath()))])
    if h.parent is not None:
        joint.CreateBody0Rel().SetTargets([prim.GetPath().GetParentPath().AppendChild(h.parent)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(a) - float(b) for a, b in zip(h.pivot, h.parent_pos))))
    else:
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in h.pivot)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(float(a) - float(b) for a, b in zip(h.pivot, spec.pos))))
    if h.frame_rot is not None:
        q = Gf.Quatf(*(float(v) for v in h.frame_rot))
        joint.CreateLocalRot0Attr(q)
        joint.CreateLocalRot1Attr(q)
    if h.local0 is not None:
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in h.local0)))
    if h.local1 is not None:
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(*(float(v) for v in h.local1)))
    if h.rot0 is not None:
        joint.CreateLocalRot0Attr(Gf.Quatf(*(float(v) for v in h.rot0)))
    if h.rot1 is not None:
        joint.CreateLocalRot1Attr(Gf.Quatf(*(float(v) for v in h.rot1)))
    if not h.enabled:
        joint.CreateJointEnabledAttr(False)
    if h.lower_deg > -1e5 or h.upper_deg < 1e5:      # beyond that: a free-spinning axle (a wheel)
        joint.CreateLowerLimitAttr(float(h.lower_deg))
        joint.CreateUpperLimitAttr(float(h.upper_deg))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(h.stiffness))
    drive.CreateTargetPositionAttr(float(h.target_deg))
    drive.CreateDampingAttr(float(h.damping))
    if h.friction > 0:
        from pxr import PhysxSchema
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr(float(h.friction))


def _add_anchor(prim, spec: Part) -> None:
    """Pin the body to the world with a fixed joint that breaks under load."""
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    anchor = spec.anchor
    joint = UsdPhysics.FixedJoint.Define(prim.GetStage(), f"{prim.GetPath()}/anchor_joint")
    # body0 unset anchors the joint to the world, so its frame is given in stage coordinates --
    # this scene is built with one environment whose origin is the world origin.
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(prim.GetPath()))])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in spec.pos)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateBreakForceAttr(float(anchor.break_force))
    joint.CreateBreakTorqueAttr(float(anchor.break_torque))
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())


def _piece_spawner(spec: Part):
    """A spawner that makes the root piece, then welds every extra piece under the same prim, so the
    whole ``Part`` is one rigid body with a compound collider, and adds its spring joint if it has one."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone

    root_func = _raw_spawner(spec.shape)

    @clone
    def spawn(prim_path: str, cfg, *args, **kwargs):
        prim = root_func(prim_path, cfg, *args, **kwargs)
        for i, piece in enumerate(spec.pieces):
            body = {
                "visual_material": _look(
                    sim_utils,
                    piece.color if piece.color is not None else spec.color,
                    piece.material if piece.material is not None else spec.material,
                    piece.opacity if piece.opacity is not None else spec.opacity,
                    piece.emissive if piece.emissive is not None else spec.emissive,
                )
            }
            if piece.collide:
                body["collision_props"] = _collision_props(spec)
                body["physics_material"] = _physics_material(sim_utils, piece, spec)
            sub = _mesh_cfg(sim_utils, piece.shape, piece.size, piece.axis, **body)
            _raw_spawner(piece.shape)(
                f"{prim_path}/piece{i}",
                sub,
                translation=tuple(piece.offset),
                orientation=_euler_xyzw(piece.rot),
            )
            if piece.hidden:
                _hide(f"{prim_path}/piece{i}")
        if spec.hide_root:
            _hide(f"{prim_path}/geometry")
        for vm in spec.vmeshes:
            _add_vmesh(prim_path, *vm)
        if spec.com is not None:
            from pxr import Gf, UsdPhysics

            UsdPhysics.MassAPI.Apply(prim).CreateCenterOfMassAttr(
                Gf.Vec3f(*(float(v) for v in spec.com))
            )
        if spec.free:
            from pxr import PhysxSchema
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if spec.name == 'mallet':
                # Keep native support impulses observable throughout the hands-free hold.
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.)
        if spec.spring is not None:
            _add_spring(prim, spec)
        if spec.anchor is not None:
            _add_anchor(prim, spec)
        if spec.hinge is not None:
            _add_hinge(prim, spec)
        if spec.hinge2 is not None:
            _add_hinge(prim, spec, spec.hinge2, "hinge2_joint")
        return prim

    return spawn


def _hide(path: str) -> None:
    """Keep a collider out of the render (guide purpose): physics still uses it."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)


def _add_vmesh(prim_path, name, pts, tris, color, material=None, opacity=1.0, uv=None) -> None:
    """A smooth-shaded visual triangle mesh under the body's prim, with its own material; no collision."""
    import isaaclab.sim as sim_utils
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    P = np.asarray(pts, float)
    T = np.asarray(tris, int)
    N = np.zeros_like(P)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    import re
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not name[0].isalpha():
        name = "m_" + name
    path = f"{prim_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in P]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(T)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for t in T for i in t]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in N]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))
    mat_path = f"{prim_path}/{name}_look"
    if isinstance(material, str) and material.startswith("tex:"):
        # printed artwork: a UsdPreviewSurface whose diffuse colour reads an image through the mesh's st
        from pxr import Sdf, UsdShade
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                               UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in np.asarray(uv, float)]))
        mat = UsdShade.Material.Define(stage, mat_path)
        surf = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, f"{mat_path}/st_reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{mat_path}/image")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(material[4:])
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        return
    look = _look(sim_utils, color, material, float(opacity))
    look.func(mat_path, look)
    sim_utils.bind_visual_material(path, mat_path)


def _build_part(spec: Part):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

    body: dict = {
        "visual_material": _look(sim_utils, spec.color, spec.material, spec.opacity, spec.emissive),
        "physics_material": _physics_material(sim_utils, spec),
    }
    collision = _collision_props(spec)
    if collision is not None:
        body["collision_props"] = collision
    if spec.free:
        body["rigid_props"] = (PhysxRigidBodyPropertiesCfg(max_depenetration_velocity=float(spec.max_depenetration))
                               if spec.max_depenetration is not None else PhysxRigidBodyPropertiesCfg())
        body["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)
    cfg = _mesh_cfg(sim_utils, spec.shape, spec.size, spec.axis, **body)
    if (spec.pieces or spec.spring is not None or spec.com is not None or spec.anchor is not None
            or spec.hinge is not None or spec.hinge2 is not None or spec.vmeshes or spec.hide_root):
        cfg.func = _piece_spawner(spec)
    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
        spawner_cfg=cfg,
        initial_pose=Pose(
            position_xyz=tuple(spec.pos),
            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2)),
        ),
    )


def install() -> None:
    """Install every patch this task needs. Safe to call more than once.

    ``patch_scene`` also repairs Arena's asset root when the scene module loads, so a task that uses
    a library ``Asset`` does not have to remember to ask for it.
    """
    patch_scene()
