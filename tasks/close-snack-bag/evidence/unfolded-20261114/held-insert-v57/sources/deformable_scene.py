"""Unfolded, sewn film pouch in a YAM scene; experimental RSS backend.

No particle is attached to the robot. Material values are an uncalibrated
equivalent laminate. This does not replace the recorded rigid Arena task.
"""
from dataclasses import dataclass,field
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


def configure_rigid_model(model,ready_start,structural_fk=False,finger_friction=None,contact_ke=1e6,arm_gain_scale=1.,finger_contact_ke=None):
    if arm_gain_scale!=1.:
        ke=model.joint_target_ke.numpy().copy();kd=model.joint_target_kd.numpy().copy()
        starts=model.joint_qd_start.numpy()
        for j,label in enumerate(model.joint_label):
            if "yam_bimanual" in str(label) and "finger" not in str(label):
                d=int(starts[j]);ke[d]*=arm_gain_scale;kd[d]*=np.sqrt(arm_gain_scale)
        model.joint_target_ke.assign(ke);model.joint_target_kd.assign(kd)
    if ready_start:
        q=model.joint_q.numpy().copy()
        starts=model.joint_q_start.numpy()
        for j,label in enumerate(model.joint_label):
            if label.endswith("left_finger"):
                q[int(starts[j])]=.0045
            elif label.endswith("right_finger"):
                q[int(starts[j])]=-.0045
        model.joint_q.assign(q)
    if structural_fk:
        from finite_gripper_drive import configure
        configure(model)
        newton.eval_fk(model,model.joint_q,model.joint_qd,model)
    model.shape_material_ke.assign(np.full(model.shape_count,contact_ke,np.float32))
    if finger_contact_ke is not None:
        if finger_contact_ke<=0:raise ValueError("positive finger contact stiffness required")
        values=model.shape_material_ke.numpy().copy()
        for shape,body in enumerate(model.shape_body.numpy()):
            if body>=0 and "finger" in str(model.body_label[body]):values[shape]=finger_contact_ke
        model.shape_material_ke.assign(values)
    if finger_friction is not None:
        mu=model.shape_material_mu.numpy().copy()
        for shape,body in enumerate(model.shape_body.numpy()):
            if body>=0 and "finger" in str(model.body_label[body]):mu[shape]=finger_friction
        model.shape_material_mu.assign(mu)


@dataclass(frozen=True,kw_only=True)
class RigidSetupSpec:
    entity:str="snack_rigid_setup"
    name:str="rigid_setup"
    ready_start:bool=True
    finger_friction:float|None=None
    structural_fk:bool=False
    contact_ke:float=1e6
    finger_contact_ke:float|None=None
    arm_gain_scale:float=1.
    sim:object=field(default_factory=lambda:rss.SimRequirements(domain="articulated"))


@rss.register_entity("snack_rigid_setup")
class RigidSetupEntity(rss.Entity):
    domain="articulated"

    def build(self,builder,episode):
        pass

    def apply_model_materials(self,model):
        configure_rigid_model(model,self.spec.ready_start,self.spec.structural_fk,
                              self.spec.finger_friction,self.spec.contact_ke,self.spec.arm_gain_scale,self.spec.finger_contact_ke)


@dataclass(frozen=True, kw_only=True)
class PouchSpec(PaperSpec):
    entity: str = "snack_unfolded_pouch"
    ready_start: bool = False
    structural_fk: bool = False
    finger_friction:float|None=None
    contact_ke:float=1e6
    finger_contact_ke:float|None=None
    arm_gain_scale:float=1.


@rss.register_entity("snack_unfolded_pouch")
class PouchEntity(EntityPaper):
    domain = "paper"

    def apply_model_materials(self, model):
        super().apply_model_materials(model)
        # Newton 1.7 VBD requires body_q and joint_q to describe the SAME
        # structural rest pose before solver construction (not just State FK).
        configure_rigid_model(model,self.spec.ready_start,self.spec.structural_fk,
                              self.spec.finger_friction,self.spec.contact_ke,self.spec.arm_gain_scale,self.spec.finger_contact_ke)

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


def make_environment(embodiment="yam_bimanual", device="cuda:0", *, cells=(40, 32), coupling="one_way", ready_start=False, with_clip=False, material_profile="derived", control_hz=60, side_access=False, physics_hz=None, rigid_cpu=False,rigid_only=False,workspace="original",clip_style="standard",support_extension=0.,board_friction=1.,finger_friction=None,self_filter_rings=0,self_radius=.0005,rebase_z=False,rigid_contact_ke=None,body_particle_buffer=None,board_skin_mm=0.,rigid_contact_threads=4,rigid_soft_dat=True,proxy_iterations=2,tool_rack_style="wide",proxy_mass_scale=1.,arm_gain_scale=1.,tool_parking_shift_x=0.,rigid_contact_model="alm",finger_contact_ke=None):
    from vbd_parallel import configure as configure_parallel
    contact_parallel=configure_parallel(rigid_contact_threads)
    if not 0.<=tool_parking_shift_x<=.06:raise ValueError("parking shift must be 0..6 cm")
    if not 1.<=arm_gain_scale<=20.:raise ValueError("arm gain scale must be 1..20")
    if tool_rack_style not in ("wide","end_access"):raise ValueError("unknown tool rack style")
    if embodiment != "yam_bimanual":
        raise ValueError("This research scene currently supports yam_bimanual only")
    if not 0<=board_skin_mm<=2.:raise ValueError("board skin must be 0..2 mm")
    contact_ke=rigid_contact_ke if rigid_contact_ke is not None else (1e4 if coupling=="unified" else 1e6)
    particle_buffer=body_particle_buffer or (4096 if coupling=="unified" else 16384)
    if contact_ke<=0 or particle_buffer<=0:raise ValueError("positive contact stiffness and capacity required")
    if not 0<=support_extension<=.014:
        raise ValueError("support extension must stay within the reset finger clearance (0..14 mm)")
    if board_friction<0 or (finger_friction is not None and finger_friction<0):
        raise ValueError("contact friction cannot be negative")
    if workspace not in ("original","raised_forward","raised_flat"):
        raise ValueError("unknown workspace")
    if workspace!="original" and not ready_start:
        raise ValueError("raised_forward currently requires the authored ready-start reset")
    shift={"original":0.,"raised_forward":.08,"raised_flat":.12}[workspace]
    height={"original":.10,"raised_forward":.20,"raised_flat":.24}[workspace] if ready_start else .04
    # 1 GPa is a comparison hypothesis for a laminate, not a measured
    # bag. It remains below the 2.507 GPa MD value in the cited BOPP sheet.
    # Retaining rest-angle changes represents irreversible crease memory.
    film=FilmMaterial(young=1e9 if material_profile=="laminate" else 1e8,
                      recovery_per_second=0. if material_profile in ("permanent","laminate") else .1)
    coeff = film.coefficients(min(.2 / cells[0], .16 / cells[1]))
    material = rss.replace(
        PAPER, zero_rest_angle=False, thickness=80e-6, areal_density=coeff["areal_density"],
        tri_ke=coeff["tri_ke"], tri_ka=coeff["tri_ka"], edge_ke=coeff["edge_ke"],
        particle_radius=.0003, yield_angle=coeff["yield_angle"], plastic_recovery=coeff["recovery"],
        friction=.5, contact_ke=1e4)
    if material_profile=="paper_matched":
        # Numerical bending reference from the pinned paper/YAM example,
        # at half its cell width. NOT a measured snack-laminate calibration.
        material=rss.replace(material, tri_ke=1e4, tri_ka=1e4,
                             edge_ke=6., yield_angle=.2)
    elif material_profile not in ("derived","permanent","laminate"):
        raise ValueError("unknown material profile")
    # The proven upstream paper/YAM path provides force feedback along the
    # fingers' slides. General two-way clip/film coupling is a later gate.
    coupling_spec = (rss.coupler("one_way", contact_forces=True, slide_feedback=True)
                     if coupling == "one_way" else rss.coupler("proxy", iterations=proxy_iterations, proxy_joints=coupling!="hybrid",
                                      proxy_relaxation=.5 if coupling=="hybrid" else 1.,mass_scale=proxy_mass_scale))
    physics_rate=physics_hz or (2400 if with_clip else 1200)
    if physics_rate<control_hz or physics_rate%control_hz:
        raise ValueError("physics_hz must be a positive multiple of control_hz")
    physics = rss.replace(
        rss.GENERIC, control_hz=float(control_hz), substeps=physics_rate//control_hz,
        coupling=coupling_spec,
        collision=rss.CollisionSpec(full_surface_contact=True,
            contact_matching="latest" if coupling in ("unified","hybrid") else "disabled"))
    if rigid_contact_model not in ("alm","soft"):
        raise ValueError("unknown rigid contact model")
    dat_enabled=coupling=="unified" and rigid_soft_dat and not rigid_only
    backend = rss.backend(
        "unified_vbd_solver.OwnedVBD" if dat_enabled else "vbd", iterations=30,
        particle_enable_self_contact=not rigid_only,rigid_soft_enable_dat=dat_enabled, friction_epsilon=1e-3,
        rigid_compliant_alm=rigid_contact_model=="alm",rigid_contact_hard=rigid_contact_model=="alm",
        rigid_contact_history=coupling in ("unified","hybrid"),
        rigid_soft_contact_use_log_barrier=coupling in ("unified","hybrid"),
        rigid_soft_dat_use_interval_arithmetic=dat_enabled,
        particle_self_contact_radius=self_radius, particle_self_contact_margin=.002,
        particle_topological_contact_filter_threshold=self_filter_rings,
        rigid_body_contact_buffer_size=2048, rigid_body_particle_contact_buffer_size=particle_buffer,
        particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128)
    physics = physics.with_backend("*", backend)
    if coupling == "unified":
        physics = physics.with_backend("articulated", backend)
    if rigid_cpu or (rigid_only and coupling=="one_way"):
        if coupling!="one_way":
            raise ValueError("Native MuJoCo control requires the one_way research path")
        physics=physics.with_backend("articulated",rss.backend("mujoco",use_mujoco_cpu=rigid_cpu))
    cell_name="bench_0728"
    z_offset=-(.728+height) if rebase_z else 0.
    if rebase_z:
        # Coordinate change only: put the thin film near z=0 so float32
        # can resolve nanometre-scale compliant contact deflections.
        from robosimstudio.scene.registry import get_cell,register_cell
        base=get_cell("bench_0728")
        cell_name=f"snack_rebased_{workspace}_{int(round(height*1000))}"
        statics=tuple(rss.replace(item,xform=(*item.xform[:2],item.xform[2]+z_offset,*item.xform[3:]))
                      for item in base.statics)
        candidate=rss.replace(base,name=cell_name,ground_z=base.ground_z+z_offset,
                              surface_z=base.surface_z+z_offset,statics=statics,
                              provenance="bench_0728 translated vertically; physical dimensions and relative poses unchanged")
        try:get_cell(cell_name)
        except KeyError:register_cell(candidate)
    scene = rss.Scene(cell=cell_name, physics=physics, name="snack_unfolded")
    grip = rss.ActuationSpec(target_ke=1e5, target_kd=20., effort_limit=10., joint_damping=100.)
    robot = get_embodiment(embodiment)
    if ready_start:
        import json
        filename={"original":"yam_ready_pose.json","raised_forward":"yam_raised_forward_pose.json",
                  "raised_flat":"yam_raised_flat_pose.json"}[workspace]
        ready = json.loads((Path(__file__).parent/"experiments/assets"/filename).read_text())
        robot = rss.replace(robot, reset_joint_q=True, arms=tuple(rss.replace(
            arm, home_q=tuple(ready["home"][arm.name]), home_gripper_q=(.0045,)) for arm in robot.arms))
    scene.add(rss.robot(robot, gripper=grip))
    # The board supports the flat mouth centrally. Its 120 mm width leaves
    # 20 mm of film on either side accessible to the fingers from below.
    half = (.085, .12, height/2) if ready_start else (.101, .06, .02)
    at = (.055+shift, 0, height/2) if ready_start else (.039, 0, .02)
    if ready_start and not side_access:
        half=(half[0]+support_extension/2,*half[1:])
        at=(at[0]-support_extension/2,*at[1:])
    if ready_start and side_access:
        # Keep the filled-body area fully supported. Only the 30 mm neck
        # extension is narrower, leaving side access for the lower pad.
        half,at=(.07,.12,height/2),(.07+shift,0,height/2)
        scene.add(rss.slab(name="mouth_support",half=(.015,.06,height/2),
                           at=(-.015+shift,0,height/2),color=(.64,.47,.30),friction=board_friction))
    skin=board_skin_mm/1000.
    if skin:
        # Same closed outer solid, split into a base and thin top collider.
        # With a rebased world, top-surface arithmetic stays near zero.
        if skin>=2*half[2]:raise ValueError("skin must be thinner than the board")
        scene.add(rss.slab(name="board_base",half=(*half[:2],half[2]-skin/2),
            at=(*at[:2],at[2]-skin/2),color=(.64,.47,.30),friction=board_friction))
        scene.add(rss.slab(name="board",half=(*half[:2],skin/2),
            at=(*at[:2],at[2]+half[2]-skin/2),color=(.64,.47,.30),friction=board_friction))
    else:
        scene.add(rss.slab(name="board", half=half, at=at,
                           color=(.64, .47, .30), friction=board_friction))
    if rigid_only:
        scene.add(RigidSetupSpec(ready_start=ready_start,finger_friction=finger_friction,
                                structural_fk=coupling=="unified",contact_ke=contact_ke,arm_gain_scale=arm_gain_scale,finger_contact_ke=finger_contact_ke))
    else:
        scene.add(PouchSpec(name="pouch", pose=(.14+shift, 0, height+.0237), quat=(0, 0, 1, 0),
                            cells=cells, size=(.2, .16), material=material,
                            color=(.96, .58, .08), settle_frames=0, ready_start=ready_start,
                            structural_fk=coupling=="unified",finger_friction=finger_friction,contact_ke=contact_ke,arm_gain_scale=arm_gain_scale,finger_contact_ke=finger_contact_ke))
    if with_clip:
        if coupling not in ("one_way","unified","hybrid"):
            raise ValueError("Clip contact requires one_way or fully unified VBD")
        from deformable_clip import ClipSpec
        scene.add(rss.slab(name="tool_base",half=(.10,.07,.025),
                           at=(tool_parking_shift_x,-.30,.025),color=(.64,.47,.30),friction=.6))
        # A passive parking support under the long handles. Without it the
        # free upright clip tips backward before the robot reaches it.
        scene.add(rss.slab(name="tool_stand",half=(.012 if tool_rack_style=="end_access" else .035,.050,.019),
                           at=((- .020 if tool_rack_style=="end_access" else -.03)+tool_parking_shift_x,-.30,.069),color=(.64,.47,.30),friction=.6))
        for side in (-1,1):
            scene.add(rss.slab(name=f"tool_rail_{side}",half=(.012 if tool_rack_style=="end_access" else .035,.002,.002),
                at=((- .020 if tool_rack_style=="end_access" else -.03)+tool_parking_shift_x,-.30+side*.047,.090),color=(.64,.47,.30),friction=.6))
        tool_spec=ClipSpec(at=(.02+tool_parking_shift_x,-.30,.102),style=clip_style,unified_spring=coupling in ("unified","hybrid"))
        if coupling=="hybrid":
            tool_spec=rss.replace(tool_spec,entity="snack_film_clip",sim=rss.SimRequirements(
                domain="paper",contacts=rss.contacts_with("surface","objects","articulated","robot","paper","cloth")))
        scene.add(tool_spec)
    env = scene.build(device=device, control_mode="joint_position")
    if rigid_cpu:
        env.pipeline.use_cuda_graph=False
    if with_clip and not rigid_only and coupling=="one_way":
        from tool_feedback import ToolFeedback
        reaction = ToolFeedback(env.pipeline.program.solver)
        env.pipeline.hooks.add("pre_solve", reaction.before_solve, name="clip-full-wrench", cadence=1)
        env.pipeline.invalidate_graph()
        env.snack_tool_feedback = reaction
    if coupling=="hybrid":
        if not rigid_only:
            from hybrid_contact import install as install_hybrid
            env.snack_hybrid_readout=install_hybrid(env)
        from unified_contact import install_joint_encoder
        from passive_clip_spring import PassiveClipSpring
        install_joint_encoder(env)
        tool=env.scene.by_name["clip"]
        spring=PassiveClipSpring(env,*tool.bodies,tool.spring_stiffness,tool.spring_damping)
        env.pipeline.hooks.add("pre_solve",spring.before_solve,name="passive-clip-spring",cadence=1)
        env.snack_passive_spring=spring
        env.pipeline.invalidate_graph()
    if coupling=="unified":
        if dat_enabled:
            from unified_vbd_solver import install_contact_authority
            install_contact_authority(env)
        from unified_contact import install,install_joint_encoder
        if not rigid_only:env.snack_unified_readout=install(env)
        install_joint_encoder(env)
        from finite_gripper_drive import FiniteGripperDrive
        gripper=FiniteGripperDrive(env)
        env.pipeline.hooks.add("pre_solve",gripper.before_solve,name="finite-gripper-drive",cadence=1)
        env.snack_finite_gripper=gripper
        from robot_gravity_compensation import RobotGravityCompensation
        compensation=RobotGravityCompensation(env)
        env.pipeline.hooks.add("pre_solve",compensation.before_solve,name="authored-robot-gravcomp",cadence=1)
        env.snack_robot_gravcomp=compensation
        env.pipeline.invalidate_graph()
        if with_clip:
            from passive_clip_spring import PassiveClipSpring
            tool=env.scene.by_name["clip"]
            spring=PassiveClipSpring(env,*tool.bodies,tool.spring_stiffness,tool.spring_damping)
            env.pipeline.hooks.add("pre_solve",spring.before_solve,name="passive-clip-spring",cadence=1)
            env.snack_passive_spring=spring
            env.pipeline.invalidate_graph()
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
    env.snack_arm_gain_scale=arm_gain_scale
    env.snack_proxy_iterations=proxy_iterations
    env.snack_contact_methods=dict(rigid_model=rigid_contact_model,rigid_history=coupling in ("unified","hybrid"),
        rigid_soft_log_barrier=coupling in ("unified","hybrid"),interval_dat=dat_enabled)
    env.snack_rigid_soft_dat=dat_enabled
    env.snack_contact_parallel=contact_parallel
    env.snack_board_skin_mm=board_skin_mm
    env.snack_world_z_offset=z_offset
    env.snack_board_height = height
    env.snack_ready_start = ready_start
    env.snack_with_clip = with_clip
    env.snack_material_parameters=dict(profile=material_profile,**vars(film),**coeff,
        particle_radius_m=.0003,self_contact_interaction_m=self_radius)
    env.snack_material_parameters.update(tri_ke=material.tri_ke,tri_ka=material.tri_ka,
        edge_ke=material.edge_ke,yield_angle=material.yield_angle,
        plastic_recovery=material.plastic_recovery)
    env.snack_material_parameters.update(board_shape_friction=board_friction,finger_shape_friction=finger_friction)
    env.snack_material_parameters.update(rigid_contact_ke=contact_ke,finger_contact_ke=finger_contact_ke,body_particle_buffer=particle_buffer)
    env.snack_material_parameters["self_contact_filter_rings"]=self_filter_rings
    if not rigid_only:
        mass=env.model.particle_mass.numpy()[env.scene.by_name["pouch"].particles]
        env.snack_material_parameters.update(actual_pouch_mass_kg=float(mass.sum()),
            zero_mass_pouch_particles=int((mass==0).sum()),
            model_gravity=np.asarray(env.model.gravity.numpy()).tolist())
    env.snack_rigid_backend="native_robot_vbd_tool_proxy" if coupling=="hybrid" else "unified_vbd" if coupling=="unified" else ("mujoco_native_cpu" if rigid_cpu else "mujoco_warp_gpu")
    env.snack_coupling=coupling
    env.snack_rigid_only=rigid_only
    env.snack_workspace=workspace
    env.snack_tool_parking_shift_x=tool_parking_shift_x
    env.snack_tool_rack_style=tool_rack_style
    env.snack_clip_style=clip_style
    support_name="mouth_support" if ready_start and side_access else "board"
    support_id=next(i for i,label in enumerate(env.model.shape_label) if str(label)==support_name)
    env.snack_support_front_m=float(env.model.shape_transform.numpy()[support_id,0]-env.model.shape_scale.numpy()[support_id,0])
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
    m = env.model
    np.savez_compressed(output/"colliders.npz", body=m.shape_body.numpy(), kind=m.shape_type.numpy(),
                        scale=m.shape_scale.numpy(), local=m.shape_transform.numpy(),
                        flags=m.shape_flags.numpy(),mu=m.shape_material_mu.numpy(),
                        margin=m.shape_margin.numpy(),
                        contact_ke=m.shape_material_ke.numpy(),contact_kd=m.shape_material_kd.numpy(),
                        particle_mass=m.particle_mass.numpy(),particle_inv_mass=m.particle_inv_mass.numpy())
