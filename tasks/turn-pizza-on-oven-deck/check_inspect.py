"""Exercise the Inspect Robots reset/observe/step interface without a model call."""

from pathlib import Path

import numpy as np
from inspect_robots import Scene
from inspect_robots.types import Action

from inspect_adapter import PizzaEmbodiment


def main():
    runner = PizzaEmbodiment(
        task=str(Path(__file__).with_name("task.py").resolve()),
        control_hz=20,
        cam_height=176,
        cam_width=320,
        object_jitter=0.0,
    )
    scene = Scene(id="fixed", instruction=runner.task.module.INSTRUCTION)
    try:
        for _ in range(2):
            obs = runner.reset(scene, seed=17)
            assert set(obs.images) == {"top_cam", "left_cam", "right_cam"}
            for image in obs.images.values():
                rgb = np.asarray(image)
                assert rgb.shape == (176, 320, 3) and rgb.std() > 2
            target = next(iter(obs.state.values())).copy()
            assert target.shape == (14,)
            for _ in range(12):
                result = runner.step(Action(data=target))
            assert result.info["score"] == 0 and runner._env.grade.fault is None
            print(
                "Fixed-scene reset, three cameras and twelve idle actions passed.",
                flush=True,
            )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
