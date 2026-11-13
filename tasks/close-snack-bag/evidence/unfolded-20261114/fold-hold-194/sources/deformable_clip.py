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
    at: tuple = (.02, -.20, .052)
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
        builder.add_mjcf(str(Path(__file__).parent/"experiments/assets/clip_spring_v2.xml"),
                         floating=True, override_root_xform=True,
                         xform=wp.transform(wp.vec3(*map(float,frame[:3])), wp.quat(*map(float,frame[3:]))))
        self.bodies = tuple(range(start_b, builder.body_count))
        self.shapes = tuple(range(start_s, builder.shape_count))
        self.coords = np.arange(start_q, len(builder.joint_q))
        for body in self.bodies:
            builder.body_label[body] = "snack_clip/"+builder.body_label[body]
        self.hinge = next(j for j in range(start_j, builder.joint_count)
                          if builder.joint_label[j].endswith("clip_hinge"))

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
