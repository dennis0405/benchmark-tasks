"""Render recorded solver coordinates only; standalone diagnostic, no robot."""
import argparse
import json
import hashlib
import os
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import cv2
import imageio.v2 as imageio
import numpy as np
# pyrender 0.1.45's shadow bounds still use the removed NumPy 1.x constant.
if not hasattr(np, 'infty'):
    np.infty = np.inf
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


def spring_mesh(start,end):
    """Schematic visualization of an ideal prismatic spring, not a collider."""
    axis=end-start;axis/=np.linalg.norm(axis)
    reference=np.array([1.,0.,0.]) if abs(axis[0])<.8 else np.array([0.,1.,0.])
    u=np.cross(axis,reference);u/=np.linalg.norm(u);v=np.cross(axis,u)
    t=np.linspace(0,1,29)
    points=start[None,:]+t[:,None]*(end-start)[None,:]
    points+=.0008*np.sin(np.pi*t)[:,None]*(np.cos(4*np.pi*t)[:,None]*u+np.sin(4*np.pi*t)[:,None]*v)
    return trimesh.util.concatenate([trimesh.creation.cylinder(radius=.00018,
        segment=np.array([a,b]),sections=5) for a,b in zip(points[:-1],points[1:])])


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
    spring_anchors=[]
    if result.get('compliant_jaw'):
        initial=data['b_00000'];upper=shapes['upper']
        parent_rotation=Rotation.from_quat(initial[upper,3:])
        frame_id=next(sid for sid,owner in enumerate(shapes['shape_body'])
            if owner==upper and shapes['shape_scale'][sid][1]>.045)
        ceiling=shapes['shape_local'][frame_id][2]-shapes['shape_scale'][frame_id][2]
        for owner,label in enumerate(shapes['body_labels']):
            if '/pad_' in label:
                local=parent_rotation.inv().apply(initial[owner,:3]-initial[upper,:3])
                spring_anchors.append((owner,np.array([local[0],local[1],ceiling])))
    renderer = pyrender.OffscreenRenderer(640, 600)
    keys = sorted(k for k in data.files if k.startswith('q_'))
    if [int(key[2:]) for key in keys] != list(range(len(keys))):
        raise ValueError('Video timing requires --record-every 1, with initial state and every control frame')
    # All frames come from recorded positions; 2x slow playback, no interpolation.
    writer = imageio.get_writer(str(output), fps=30, codec='libx264', quality=8,
                               ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    film_node = None
    spring_node=None
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
        if spring_node is not None:scene.remove_node(spring_node)
        if spring_anchors:
            meshes=[]
            for owner,anchor in spring_anchors:
                start=Rotation.from_quat(bodies[shapes['upper'],3:]).apply(anchor)+bodies[shapes['upper'],:3]
                end=Rotation.from_quat(bodies[owner,3:]).apply([0,0,.003])+bodies[owner,:3]
                meshes.append(spring_mesh(start,end))
            spring_node=scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(meshes),
                material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.7,.73,.76,1],metallicFactor=.8,roughnessFactor=.3)))
        views = []
        for eye, target in (([.37, -.32, .32], [.11, 0, .014]),
                            ([.18, -.27, .065], [.16, 0, .012])):
            scene.set_pose(camera, camera_pose(eye, target))
            rgb, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            views.append(rgb)
        canvas = np.full((720, 1280, 3), 245, dtype=np.uint8)
        canvas[65:665] = np.concatenate(views, axis=1)
        title='DEFORMABLE FILM + FIVE SPRING PADS' if spring_anchors else 'DEFORMABLE FILM + FREE SPRING CLIP'
        lines = [(title, (20, 28)),
                 ('Recorded Newton physics | empty pouch, pre-folded reset | no robot / no pose locking', (20, 54))]
        if frame:
            row = trace[frame-1]
            sub = result['substeps']
            slot = frame*sub
            label = f't = {frame/60:.2f}s | {slot+1} physics states checked | jaw = {np.rad2deg(row["jaw_angle_rad"]):.1f}deg | replay 0.5x'
        else:
            label = 't = 0.00s | open clip around pre-folded mouth'
            slot = 0
        geometry_labels=[]
        for key,label_name in (('clip_intrusions','clip/film'),('self_crossings','film/self'),
                               ('clip_self_intrusions','clip/self'),('ground_intrusions','table')):
            value=str(int(np.count_nonzero(counts[key][:slot+1]))) if key in counts else 'n/a'
            geometry_labels.append(f'{label_name}: {value}')
        lines.append((label, (20, 686)))
        lines.append(('Intersection states | '+' | '.join(geometry_labels), (20, 710)))
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
        if frame in (0, 60, 180, 300) or key == keys[-1]:
            imageio.imwrite(str(args.run/f'frame-{frame:05d}.png'), canvas)
    writer.close()
    renderer.delete()
    (args.run/'render-manifest.json').write_text(json.dumps({
        'renderer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recording_sha256': hashlib.sha256((args.run/'pouch-snapshots.npz').read_bytes()).hexdigest(),
        'video_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'rendered_control_frames': [int(key[2:]) for key in keys[::2]],
        'output_fps': 30, 'copies_per_source_frame': 2, 'playback_speed': .5,
        'position_interpolation': False, 'physics_modified_for_rendering': False,
        'spring_links_schematic_noncolliding':bool(spring_anchors)}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
