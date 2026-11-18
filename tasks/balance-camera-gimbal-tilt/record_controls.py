"""Record real passive releases; each column is a different fixed rail adjustment."""
import argparse
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from probe import FILE

def record(path):
    model = mujoco.MjModel.from_xml_path(str(FILE))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.008, 0, .118)
    camera.azimuth, camera.elevation, camera.distance = 105, -15, .46
    width, height, fps = 384, 288, 25
    with mujoco.Renderer(model, height, width) as renderer:
        encoder = subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width*3}x{height}','-r',str(fps),'-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        try:
            for angle in (-30, 30):
                trials = []
                for offset in (.015,-.019,-.045):
                    data = mujoco.MjData(model)
                    data.qpos[:]=[math.radians(angle),offset,0]
                    mujoco.mj_forward(model,data)
                    trials.append(data)
                for frame in range(3*fps+1):
                    views=[]
                    for name, data in zip(('Front-heavy','Balanced','Rear-heavy'),trials):
                        renderer.update_scene(data,camera)
                        im=Image.fromarray(renderer.render())
                        draw=ImageDraw.Draw(im)
                        draw.rectangle((0,0,width,32),fill=(24,29,34))
                        draw.text((8,5),f'{name}  |  pitch {math.degrees(data.qpos[0]):+.1f} deg',fill='white')
                        draw.text((8,18),f'Free release from {angle:+d} deg  |  {frame/fps:.2f} s',fill='white')
                        views.append(np.asarray(im))
                    encoder.stdin.write(np.concatenate(views,axis=1).tobytes())
                    for data in trials:
                        for _ in range(round(1/(fps*model.opt.timestep))):mujoco.mj_step(model,data)
        finally:
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('ffmpeg encoding failed')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('path',type=Path)
    record(parser.parse_args().path)
