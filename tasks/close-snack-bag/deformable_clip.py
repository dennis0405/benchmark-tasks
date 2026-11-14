"""A free spring clip for the unfolded-pouch research scene."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warp as wp
import robosimstudio as rss
import numpy as np


@dataclass(frozen=True, kw_only=True)
class ClipSpec:
    entity: str = "snack_free_clip"
    name: str = "clip"
    style: str = "standard"
    unified_spring:bool=False
    at: tuple = (.02, -.30, .102)
    quat: tuple = (.7071067811865475, 0., 0., .7071067811865475)
    sim: Any = field(default_factory=lambda:rss.SimRequirements(
        domain="articulated", contacts=rss.contacts_with("surface", "objects", "paper", "cloth")))


@rss.register_entity("snack_free_clip")
class ClipEntity(rss.Entity):
    domain = "articulated"

    def build(self, builder, episode):
        del episode
        start_b, start_s, start_j = builder.body_count, builder.shape_count, builder.joint_count
        start_q = len(builder.joint_q)
        frame = self.placed((*self.spec.at, *self.spec.quat))
        filenames={"standard":"clip_spring_v2.xml","thin_lower":"clip_thin_lower.xml","thin_strong":"clip_thin_strong.xml","thin_broad":"clip_thin_broad.xml","thin_aligned":"clip_thin_aligned.xml","thin_compact":"clip_thin_compact.xml"}
        asset=Path(__file__).parent/"experiments/assets"/filenames[self.spec.style]
        builder.add_mjcf(str(asset),
                         floating=True, override_root_xform=True,
                         xform=wp.transform(wp.vec3(*map(float,frame[:3])), wp.quat(*map(float,frame[3:]))))
        self.bodies = tuple(range(start_b, builder.body_count))
        self.shapes = tuple(range(start_s, builder.shape_count))
        self.coords = np.arange(start_q, len(builder.joint_q))
        for body in self.bodies:
            builder.body_label[body] = "snack_clip/"+builder.body_label[body]
        self.hinge = next(j for j in range(start_j, builder.joint_count)
                          if builder.joint_label[j].endswith("clip_hinge"))
        import xml.etree.ElementTree as ET
        joint=ET.parse(asset).getroot().find("worldbody/body/body/joint")
        self.spring_stiffness=float(joint.get("stiffness","0"))
        self.spring_damping=float(joint.get("damping","0"))
        if self.spec.unified_spring:
            dof=builder.joint_qd_start[self.hinge]
            for name in ("joint_target_ke","joint_target_kd","joint_damping"):
                values=getattr(builder,name,None)
                if values is not None:values[dof]=0.

    def handle(self, model):
        del model
        from robosimstudio.scene.handles import EntityHandle
        return EntityHandle(name=self.name, body=self.bodies[0], bodies=self.bodies, shapes=self.shapes)

    def reset(self, pipeline, episode, **kwargs):
        del episode, kwargs
        for state in (pipeline.state_0, pipeline.state_1):
            q = state.joint_q.numpy().copy()
            q[self.coords] = self.model.joint_q.numpy()[self.coords]
            state.joint_q.assign(q)
