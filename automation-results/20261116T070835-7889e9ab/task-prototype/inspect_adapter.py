"""Construct the custom screen while preserving the standard YAM observation and action protocol."""

import atexit
import os
from yam_arena import inspect_embodiment as yam


class SieveEmbodiment(yam.YamArenaEmbodiment):
    def _ensure_sim(self, goal):
        if self._env is not None:
            return
        if self.object_jitter or self.info.control_hz != 20 or goal is not None:
            raise ValueError(
                "This fixed layout uses object_jitter=0 and control_hz=20."
            )
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if yam._APP is None:
            from isaaclab.app import AppLauncher

            kwargs = dict(
                headless=self.headless, device=self.device, enable_cameras=True
            )
            if not self.headless:
                kwargs["visualizer"] = ["kit"]
            yam._APP = AppLauncher(**kwargs).app
            atexit.register(yam._close_app_at_exit)
        import torch

        self._env = self.task.module.make_env(
            self.device, True, round(yam.SETTLE_S * self.info.control_hz)
        )
        self.task.module._active_env = self._env
        self._base = self._env.unwrapped
        self._goal = goal
        self._free_objects = self.task.free_objects
        self._torch = torch
        self._ready = torch.tensor(
            [[*yam.READY, 1.0, *yam.READY, 1.0]],
            dtype=torch.float32,
            device=self._base.device,
        )
        self._ids = {
            name: tuple(
                self._base.scene[name].find_joints(joints, preserve_order=True)[0]
                for joints in (yam.ARM_JOINTS, yam.FINGER_JOINTS)
            )
            for name in yam.ARMS
        }
