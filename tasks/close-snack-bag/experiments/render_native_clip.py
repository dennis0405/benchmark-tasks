"""Render recorded solver coordinates only; standalone diagnostic, no robot."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import cv2
import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation


def pose(values):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def camera_pose(eye, target):
    eye, target = np.array(eye), np.array(target)
    back = eye-target
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    result = np.eye(4)
    result[:3, :3] = np.stack([right, up, back], axis=1)
    result[:3, 3] = eye
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = np.load(args.run/'pouch-snapshots.npz')
    shapes = json.loads((args.run/'clip-shapes.json').read_text())
    trace = json.loads((args.run/'trace.json').read_text())
    result = json.loads((args.run/'result.json').read_text())
    hold_path = args.run/'hold-check.json'
    hold = json.loads(hold_path.read_text()) if hold_path.exists() else None
    counts = np.load(args.run/'geometry-counts.npz')
    output = args.output or args.run/'clip-film-diagnostic.mp4'
    scene = pyrender.Scene(bg_color=[.93, .95, .97, 1], ambient_light=[.5, .5, .5])
    table = trimesh.creation.box([.6, .5, .006])
    table.apply_translation([.1, 0, -.003])
    scene.add(pyrender.Mesh.from_trimesh(table, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[.72, .76, .78, 1], roughnessFactor=.9)))
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38), znear=.001, zfar=3))
    for eye in ([.25, -.2, .5], [-.15, .15, .4]):
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.2),
                  pose=camera_pose(eye, [.12, 0, 0]))
    clip_nodes = []
    for sid, owner in enumerate(shapes['shape_body']):
        if owner < 0:
            continue
        half = np.array(shapes['shape_scale'][sid])
        mesh = trimesh.creation.box(2*half)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.15, .22, .3, 1],
                    metallicFactor=.25, roughnessFactor=.35)
        clip_nodes.append((sid, owner, scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))))
    renderer = pyrender.OffscreenRenderer(640, 620)
    keys = sorted(k for k in data.files if k.startswith('q_'))
    # All frames come from recorded positions; 2x slow playback, no interpolation.
    writer = imageio.get_writer(str(output), fps=30, codec='libx264', quality=8,
                               ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    film_node = None
    for key in keys[::2]:
        frame = int(key[2:])
        if film_node is not None:
            scene.remove_node(film_node)
        film = trimesh.Trimesh(vertices=data[key], faces=data['faces'], process=False)
        # Explicit material avoids platform-dependent uint8 vertex-color handling.
        film_node = scene.add(pyrender.Mesh.from_trimesh(film, smooth=False,
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.96, .56, .08, 1],
                         metallicFactor=.05, roughnessFactor=.65, doubleSided=True)))
        bodies = data['b_'+key[2:]]
        for sid, owner, node in clip_nodes:
            scene.set_pose(node, pose(bodies[owner])@pose(shapes['shape_local'][sid]))
        views = []
        for eye, target in (([.37, -.32, .32], [.11, 0, .014]),
                            ([.18, -.27, .065], [.16, 0, .012])):
            scene.set_pose(camera, camera_pose(eye, target))
            rgb, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            views.append(rgb)
        canvas = np.full((720, 1280, 3), 245, dtype=np.uint8)
        canvas[65:685] = np.concatenate(views, axis=1)
        lines = [('DEFORMABLE FILM + FREE SPRING CLIP', (20, 28)),
                 ('Recorded Newton physics | pre-folded reset | no robot / no pose locking', (20, 54))]
        if frame:
            row = trace[frame-1]
            sub = result['substeps']
            slot = frame*sub
            intrusions = int(counts['clip_intrusions'][:slot+1].max())
            crossings = int(counts['self_crossings'][:slot+1].max())
            label = f't = {frame/60:.2f}s   jaw = {np.rad2deg(row["jaw_angle_rad"]):.1f}deg   intrusion max = {intrusions}   strict self crossing max = {crossings}'
        else:
            label = 't = 0.00s | open clip around pre-folded mouth'
        lines.append((label, (20, 708)))
        for text, location in lines:
            cv2.putText(canvas, text, location, cv2.FONT_HERSHEY_SIMPLEX,
                        .62 if location[1] != 28 else .78, (30, 40, 50), 1, cv2.LINE_AA)
        if hold and hold['passed']:
            window = hold['best_window']
            elapsed = np.clip(frame/60-window['start_s'], 0, window['hold_seconds'])
            cv2.putText(canvas, f'Validated hold window: {elapsed:.2f} / {window["hold_seconds"]:.2f}s',
                        (720, 105), cv2.FONT_HERSHEY_SIMPLEX, .6, (15, 95, 35), 2, cv2.LINE_AA)
        # Two identical output frames preserve 0.5x speed, without fabricated coordinates.
        writer.append_data(canvas)
        writer.append_data(canvas)
        if frame in (0, 60, 180, 300):
            imageio.imwrite(str(args.run/f'frame-{frame:05d}.png'), canvas)
    writer.close()
    renderer.delete()
    print(output)


if __name__ == '__main__':
    main()
