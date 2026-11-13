"""Render recorded pouch and YAM coordinates, with no interpolation."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_native_clip import pose, camera_pose


def wrapper_texture():
    """Original generic print; display only, fixed in material coordinates."""
    image=Image.new("RGB",(1024,1280),(233,139,31))
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,1023,190),fill=(218,116,23))
    for x in range(0,1024,16):
        draw.line((x,1195,x,1279),fill=(165,96,25),width=3)
    for x in (16,1008):
        draw.line((x,0,x,1279),fill=(248,182,71),width=10)
    draw.rounded_rectangle((75,360,949,650),radius=55,fill=(252,243,213))
    draw.text((512,500),"CHIPS",anchor="mm",font=ImageFont.load_default(size=170),
              fill=(159,56,25),stroke_width=2)
    draw.text((512,710),"CRISPY POTATO SNACK",anchor="mm",
              font=ImageFont.load_default(size=45),fill=(255,248,224))
    for x,y in ((320,940),(650,970),(490,875)):
        draw.ellipse((x-145,y-70,x+145,y+70),fill=(165,94,17),outline=(255,219,103),width=6)
        draw.ellipse((x-130,y-67,x+130,y+55),fill=(249,206,82))
    draw.text((512,1110),"RESEAL AFTER OPENING",anchor="mm",
              font=ImageFont.load_default(size=38),fill=(255,248,224))
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()
    data = np.load(args.run / "states.npz")
    trace = json.loads((args.run / "trace.json").read_text())
    hz=json.loads((args.run/"arguments.json").read_text()).get("hz",60)
    geometry = np.load(args.run/"render-geometry.npz")
    parts = json.loads((args.run/"render-geometry.json").read_text())
    scene = pyrender.Scene(bg_color=(.93, .95, .97, 1), ambient_light=(.85, .85, .85))
    nodes = []
    for i, part in enumerate(parts):
        mesh = trimesh.Trimesh(vertices=geometry[f"v_{i}"], faces=geometry[f"f_{i}"],
                               vertex_colors=geometry[f"c_{i}"], process=False)
        nodes.append((part, scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))))
    center = data["q_00000"].mean(axis=0)
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40), znear=.005, zfar=5))
    for eye in (center+[-.4, -.3, .8], center+[.3, .5, .6]):
        scene.add(pyrender.DirectionalLight(color=(1, 1, 1), intensity=2.5),
                  pose=camera_pose(eye, center))
    renderer = pyrender.OffscreenRenderer(640, 540)
    film_node = None
    texture_image=wrapper_texture()
    texture_image.save(args.run/"wrapper-print.png")
    q0=data["q_00000"]
    uv=np.column_stack(((q0[:,1]-q0[:,1].min())/np.ptp(q0[:,1]),
                        (q0[:,0].max()-q0[:,0])/np.ptp(q0[:,0])))
    film_material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.,1.,1.,1.),
        baseColorTexture=pyrender.Texture(source=np.asarray(texture_image),source_channels="RGB"),
        metallicFactor=.15,roughnessFactor=.45,doubleSided=True)
    keys = sorted(k for k in data.files if k.startswith("q_"))[::args.stride]
    output = args.run / "unfolded-yam-diagnostic.mp4"
    writer = imageio.get_writer(str(output), fps=hz/args.stride, codec="libx264",
                               quality=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    for key in keys:
        step = int(key[2:])
        if film_node:
            scene.remove_node(film_node)
        film = trimesh.Trimesh(vertices=data[key], faces=data["faces"], process=False)
        film.visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=texture_image)
        film_node = scene.add(pyrender.Mesh.from_trimesh(film, smooth=False,material=film_material))
        bodies = data["b_"+key[2:]]
        for part, node in nodes:
            transform = np.array(part["local"])
            if part["body"] >= 0:
                transform = pose(bodies[part["body"]]) @ transform
            scene.set_pose(node, transform)
        views=[]
        for eye, target in ((center+[.75, -.60, .65], center+[-.10, 0, .08]),
                            (center+[-.10, -.03, .40], center+[-.065, 0, 0])):
            scene.set_pose(camera, camera_pose(eye, target))
            rgb, _ = renderer.render(scene)
            views.append(rgb)
        canvas = np.full((640, 1280, 3), 245, np.uint8)
        canvas[55:595] = np.concatenate(views, axis=1)
        label = trace[step]["phase"]
        cv2.putText(canvas, "YAM + UNFOLDED FILM POUCH | contact-driven diagnostic", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 40, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={step/hz:.2f}s | phase={label} | recorded physics, 1x playback | not a completed sealing demo",
                    (20, 625), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 40, 50), 1, cv2.LINE_AA)
        writer.append_data(canvas)
        if args.stride>=30 or step in (0, 60, 300, 600, 900) or key == keys[-1]:
            imageio.imwrite(args.run/f"view-{step:05d}.png", canvas)
    writer.close()
    renderer.delete()
    (args.run/"render-manifest.json").write_text(json.dumps(dict(
        recording_sha256=hashlib.sha256((args.run/"states.npz").read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_steps=[int(key[2:]) for key in keys], fps=hz/args.stride,
        physics_modified=False,position_interpolation=False,
        wrapper_print="original procedural graphic, fixed UV; no collision effect"),indent=2))
    print(output)


if __name__ == "__main__":
    main()
