"""Unfolded, sewn film pouch in a YAM scene; experimental RSS backend.

No particle is attached to the robot. Material values are an uncalibrated
equivalent laminate. This does not replace the recorded rigid Arena task.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import newton
import warp as wp
import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import EntityPaper, PaperSpec, PAPER

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
from film_material import FilmMaterial
from pouch_geometry import open_pouch


@dataclass(frozen=True, kw_only=True)
class PouchSpec(PaperSpec):
    entity: str = "snack_unfolded_pouch"
    ready_start: bool = False
    structural_fk: bool = False


@rss.register_entity("snack_unfolded_pouch")
class PouchEntity(EntityPaper):
    domain = "paper"

    def apply_model_materials(self, model):
        super().apply_model_materials(model)
        if self.spec.ready_start:
            q = model.joint_q.numpy().copy()
            for j, label in enumerate(model.joint_label):
                if label.endswith("left_finger"):
                    q[int(model.joint_q_start.numpy()[j])] = .0025
                elif label.endswith("right_finger"):
                    q[int(model.joint_q_start.numpy()[j])] = -.0025
            model.joint_q.assign(q)
        # Newton 1.7 VBD requires body_q and joint_q to describe the SAME
        # structural rest pose before solver construction (not just State FK).
        if self.spec.structural_fk:
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        model.shape_material_ke.assign(np.full(model.shape_count, 1e6, np.float32))

    def surface(self):
        q, f, self.grid = open_pouch(
            nx=self.spec.cells[0], ny=self.spec.cells[1],
            mouth_half_gap=.0006, foldable_mouth=True)
        transition = np.clip((q[:, 0] - .12) / .02, 0, 1)
        q[:, 2] -= .022 * transition**2 * (3 - 2 * transition)
        return q, f

    def build(self, builder, episode):
        del episode
        q, f = self.surface()
        self.faces = f
        self.surface_count, self.lumping = len(q), "mesh"
        frame = self.placed((*self.spec.pose, *self.spec.quat))
        material = self.spec.material
        builder.add_cloth_mesh(
            pos=wp.vec3(*map(float, frame[:3])), rot=wp.quat(*map(float, frame[3:])),
            scale=1., vel=wp.vec3(0), vertices=[wp.vec3(*map(float, p)) for p in q],
            indices=f.ravel().tolist(), density=material.areal_density,
            tri_ke=material.tri_ke, tri_ka=material.tri_ka, tri_kd=material.tri_kd,
            edge_ke=material.edge_ke, edge_kd=material.edge_kd,
            particle_radius=material.particle_radius, label=self.name)


def make_environment(embodiment="yam_bimanual", device="cuda:0", *, cells=(40, 32), coupling="one_way", ready_start=False):
    if embodiment != "yam_bimanual":
        raise ValueError("This research scene currently supports yam_bimanual only")
    coeff = FilmMaterial().coefficients(min(.2 / cells[0], .16 / cells[1]))
    material = rss.replace(
        PAPER, zero_rest_angle=False, thickness=80e-6, areal_density=coeff["areal_density"],
        tri_ke=coeff["tri_ke"], tri_ka=coeff["tri_ka"], edge_ke=coeff["edge_ke"],
        particle_radius=.0003, yield_angle=coeff["yield_angle"], plastic_recovery=.1,
        friction=.5, contact_ke=1e4)
    # The proven upstream paper/YAM path provides force feedback along the
    # fingers' slides. General two-way clip/film coupling is a later gate.
    coupling_spec = (rss.coupler("one_way", contact_forces=True, slide_feedback=True)
                     if coupling == "one_way" else rss.coupler("proxy", iterations=2, proxy_joints=True))
    physics = rss.replace(
        rss.GENERIC, control_hz=60., substeps=20,
        coupling=coupling_spec,
        collision=rss.CollisionSpec(full_surface_contact=True))
    backend = rss.backend(
        "vbd", iterations=30, friction_epsilon=1e-3,
        rigid_compliant_alm=True,
        particle_self_contact_radius=.0003, particle_self_contact_margin=.001,
        particle_topological_contact_filter_threshold=1,
        rigid_body_contact_buffer_size=2048, rigid_body_particle_contact_buffer_size=16384,
        particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128)
    physics = physics.with_backend("*", backend)
    if coupling == "unified":
        physics = physics.with_backend("articulated", backend)
    scene = rss.Scene(cell="bench_0728", physics=physics, name="snack_unfolded")
    grip = rss.ActuationSpec(target_ke=1e5, target_kd=20., effort_limit=10., joint_damping=100.)
    robot = get_embodiment(embodiment)
    if ready_start:
        import json
        ready = json.loads((Path(__file__).parent/"experiments/assets/yam_ready_pose.json").read_text())
        robot = rss.replace(robot, reset_joint_q=True, arms=tuple(rss.replace(
            arm, home_q=tuple(ready["home"][arm.name]), home_gripper_q=(.0025,)) for arm in robot.arms))
    scene.add(rss.robot(robot, gripper=grip))
    # The board supports the flat mouth centrally. Its 120 mm width leaves
    # 20 mm of film on either side accessible to the fingers from below.
    half = (.085, .12, .05) if ready_start else (.101, .06, .02)
    at = (.055, 0, .05) if ready_start else (.039, 0, .02)
    height = .10 if ready_start else .04
    scene.add(rss.slab(name="board", half=half, at=at,
                       color=(.64, .47, .30), friction=1.))
    scene.add(PouchSpec(name="pouch", pose=(.14, 0, height+.023), quat=(0, 0, 1, 0),
                        cells=cells, size=(.2, .16), material=material,
                        color=(.96, .58, .08), settle_frames=0, ready_start=ready_start,
                        structural_fk=coupling=="unified"))
    env = scene.build(device=device, control_mode="joint_position")
    if ready_start:
        # Start-pose construction can warm the rigid solver. Restore this
        # declared episode reset once, then reset its internal state as well.
        for state in (env.pipeline.state_0, env.pipeline.state_1):
            state.joint_q.assign(env.model.joint_q)
            state.joint_qd.zero_()
            state.body_qd.zero_()
            newton.eval_fk(env.model, env.model.joint_q, env.model.joint_qd, state)
        env.pipeline.reset_solvers(flags=0)
        env.pipeline.reset_finite_flags()
        env.pipeline.invalidate_graph()
    env.snack_board_height = height
    env.snack_ready_start = ready_start
    return env


def read_object_states(env):
    entity = env.scene.by_name["pouch"]
    points = env.state.particle_q.numpy()[entity.particles].copy()
    return {"pouch": points, "bodies": env.state.body_q.numpy().copy()}


def export_render_geometry(env, output):
    """Freeze the actual model's visible meshes, not a second authored scene."""
    import json
    import trimesh
    from robosimstudio.core.render.visual import model_visual_parts
    arrays, reports = {}, []
    for i, part in enumerate(model_visual_parts(env.model, plane_extent=2.)):
        mesh = part.mesh.to_geometry() if isinstance(part.mesh, trimesh.Scene) else part.mesh
        arrays[f"v_{i}"] = np.asarray(mesh.vertices)
        arrays[f"f_{i}"] = np.asarray(mesh.faces)
        visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
        arrays[f"c_{i}"] = np.asarray(visual.vertex_colors)
        reports.append(dict(body=part.body, local=part.local.tolist(), name=part.name))
    np.savez_compressed(output / "render-geometry.npz", **arrays)
    (output / "render-geometry.json").write_text(json.dumps(reports))
