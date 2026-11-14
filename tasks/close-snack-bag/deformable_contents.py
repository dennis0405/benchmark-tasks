"""Free rigid chip surrogates inside the sewn film pouch.

Mass/contact experiment only: these chips do not yet fracture. No body is
attached to the pouch or table. Dimensions and density are uncalibrated.
"""
from dataclasses import dataclass,field
import numpy as np
import newton
import robosimstudio as rss
import warp as wp


@dataclass(frozen=True,kw_only=True)
class ContentsSpec:
    entity:str="snack_rigid_contents"
    name:str="contents"
    board_height:float=.24
    coordinate_model:str="body_only"
    sim:object=field(default_factory=lambda:rss.SimRequirements(
        domain="paper",contacts=rss.contacts_with("surface","objects","paper","cloth")))


@rss.register_entity("snack_rigid_contents")
class ContentsEntity(rss.Entity):
    domain="paper"

    def build(self,builder,episode):
        del episode
        if self.spec.coordinate_model not in ("body_only","free_joint","brittle"):raise ValueError("unknown contents coordinates")
        self.bodies=[];self.shapes=[];self.coords=[];self.initial_coordinates=[];self.initial_frames=[]
        self.chip_groups=[];self.seams=[]
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.,ke=1e4,kd=100.,mu=.6,margin=0.)
        for layer,z in enumerate((.022,.025)):
            for row,x in enumerate((.168,.222)):
                for column,y in enumerate((-.045,0.,.045)):
                    index=len(self.chip_groups)
                    angle=np.deg2rad(4.*((index%3)-1))
                    frame=self.placed((x,y,self.spec.board_height+z,0.,0.,np.sin(angle/2),np.cos(angle/2)))
                    if self.spec.coordinate_model=="brittle":
                        self.build_brittle_chip(builder,index,np.asarray(frame))
                        continue
                    # add_body() already creates a FREE joint in Newton 1.7;
                    # add_link() is the explicit link-then-joint API.
                    body=builder.add_link(xform=wp.transform(wp.vec3(*map(float,frame[:3])),
                        wp.quat(*map(float,frame[3:]))),label=f"snack_contents/chip_{index:02d}")
                    start=len(builder.joint_q)
                    initial=[]
                    if self.spec.coordinate_model=="free_joint":
                        joint=builder.add_joint_free(body,label=f"snack_contents/free_{index:02d}")
                        initial=np.asarray(builder.joint_q[start:],dtype=float)
                        if initial.shape!=(7,) or not np.allclose(initial[:3],frame[:3],atol=1e-6):
                            raise ValueError("free chip reset coordinates do not match the authored frame")
                        builder.add_articulation([joint],label=f"snack_contents/chip_{index:02d}")
                    shape=builder.add_shape_ellipsoid(body,rx=.024,ry=.018,rz=.0012,cfg=cfg,
                        color=wp.vec3(.93,.65,.17),label=f"snack_contents/chip_{index:02d}/solid")
                    self.bodies.append(body);self.shapes.append(shape)
                    self.chip_groups.append((body,))
                    self.coords.extend(range(start,len(builder.joint_q)))
                    self.initial_coordinates.extend(initial)
                    self.initial_frames.append(frame)
        self.bodies=tuple(self.bodies);self.shapes=tuple(self.shapes)
        self.coords=np.asarray(self.coords,dtype=int)
        self.initial_coordinates=np.asarray(self.initial_coordinates,dtype=np.float32)
        self.initial_frames=np.asarray(self.initial_frames,dtype=np.float32)

    def build_brittle_chip(self,builder,index,frame):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(frame[3:]);parts=[]
        cfg=newton.ModelBuilder.ShapeConfig(density=524.,ke=1e4,kd=100.,mu=.6,margin=0.)
        for half,x in enumerate((-.012,.012)):
            pose=frame.copy();pose[:3]+=rotation.apply([x,0,0])
            body=builder.add_link(xform=wp.transform(wp.vec3(*map(float,pose[:3])),
                wp.quat(*map(float,pose[3:]))),label=f"snack_contents/chip_{index:02d}/half_{half}")
            shape=builder.add_shape_box(body,hx=.012,hy=.018,hz=.0012,cfg=cfg,
                color=wp.vec3(.93,.65,.17),label=f"snack_contents/chip_{index:02d}/solid_{half}")
            parts.append(body);self.bodies.append(body);self.shapes.append(shape);self.initial_frames.append(pose)
        start=len(builder.joint_q)
        tangent=wp.quat_from_axis_angle(wp.vec3(0,1,0),np.pi/2)
        seam=builder.add_joint_rod(*parts,
            parent_xform=wp.transform(wp.vec3(.012,0,0),tangent),
            child_xform=wp.transform(wp.vec3(-.012,0,0),tangent),
            stretch_stiffness=1e4,shear_stiffness=1e4,bend_stiffness=.05,twist_stiffness=.05,
            stretch_damping=2.,shear_damping=2.,bend_damping=.0001,twist_damping=.0001,
            collision_filter_parent=False,label=f"snack_contents/chip_{index:02d}/seam")
        builder.add_articulation([seam]);self.seams.append(seam);self.chip_groups.append(tuple(parts))
        self.coords.extend(range(start,len(builder.joint_q)))
        self.initial_coordinates.extend(builder.joint_q[start:])

    def handle(self,model):
        del model
        from robosimstudio.scene.handles import EntityHandle
        return EntityHandle(name=self.name,body=self.bodies[0],bodies=self.bodies,shapes=self.shapes)

    def reset(self,pipeline,episode,**kwargs):
        del episode,kwargs
        for state in (pipeline.state_0,pipeline.state_1):
            coordinates=state.joint_q.numpy().copy()
            coordinates[self.coords]=self.initial_coordinates
            state.joint_q.assign(coordinates)
            newton.eval_fk(self.model,state.joint_q,state.joint_qd,state)
            poses=state.body_q.numpy().copy();poses[list(self.bodies)]=self.initial_frames
            state.body_q.assign(poses)
            rates=state.body_qd.numpy().copy();rates[list(self.bodies)]=0.
            state.body_qd.assign(rates)
