"""Render recorded pouch and YAM coordinates, with no interpolation."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib
import io
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_native_clip import pose, camera_pose


def wrapper_texture(cracker=False):
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
    draw.text((512,710),"CRISPY CORN SNACK" if cracker else "CRISPY POTATO SNACK",anchor="mm",
              font=ImageFont.load_default(size=45),fill=(255,248,224))
    for x,y in ((320,940),(650,970),(490,875)):
        if cracker:
            draw.rounded_rectangle((x-105,y-80,x+105,y+80),radius=12,fill=(249,206,82),outline=(165,94,17),width=7)
            for dx in (-65,0,65):
                for dy in (-45,0,45):draw.ellipse((x+dx-5,y+dy-5,x+dx+5,y+dy+5),fill=(165,94,17))
        else:
            draw.ellipse((x-145,y-70,x+145,y+70),fill=(165,94,17),outline=(255,219,103),width=6)
            draw.ellipse((x-130,y-67,x+130,y+55),fill=(249,206,82))
    draw.text((512,1110),"RESEAL AFTER OPENING",anchor="mm",
              font=ImageFont.load_default(size=38),fill=(255,248,224))
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--detail",choices=("mouth","mouth_side","mouth_front","clip"),default="mouth")
    p.add_argument("--translucent-robot-detail",action="store_true",
                   help="Show the recorded robot translucently in the labelled close-up only")
    p.add_argument("--first-step",type=int,default=0)
    p.add_argument("--last-step",type=int,default=None)
    args = p.parse_args()
    recording_bytes=(args.run/"states.npz").read_bytes()
    trace_bytes=(args.run/"trace.json").read_bytes()
    data = np.load(io.BytesIO(recording_bytes))
    trace = json.loads(trace_bytes)
    arguments=json.loads((args.run/"arguments.json").read_text());hz=arguments.get("hz",60)
    pinch_test=arguments.get("stage")=="pinch_lift"
    cracker=bool(arguments.get("contents") and arguments.get("contents_mode")=="brittle")
    geometry = np.load(args.run/"render-geometry.npz")
    parts = json.loads((args.run/"render-geometry.json").read_text())
    scene = pyrender.Scene(bg_color=(.93, .95, .97, 1), ambient_light=(.70, .70, .70))
    nodes = []
    for i, part in enumerate(parts):
        mesh = trimesh.Trimesh(vertices=geometry[f"v_{i}"], faces=geometry[f"f_{i}"],
                               vertex_colors=geometry[f"c_{i}"], process=False)
        nodes.append((part, scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))))
    has_film="q_00000" in data
    center = data["q_00000"].mean(axis=0) if has_film else np.asarray(trace[0]["clip_root_pose"][:3])+[0,.10,.06]
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40), znear=.005, zfar=5))
    for eye in (center+[-.4, -.3, .8], center+[.3, .5, .6]):
        scene.add(pyrender.DirectionalLight(color=(1, 1, 1), intensity=2.5),
                  pose=camera_pose(eye, center))
    renderer = pyrender.OffscreenRenderer(640, 540,point_size=5.)
    film_nodes = []
    tracked={}
    if pinch_test and (args.run/"result.json").exists():
        refs=json.loads((args.run/"result.json").read_text()).get("controller_diagnostics",{}).get("pinch_lift_reference",{})
        for arm,base in refs.items():
            selected=[]
            for ids,points in zip(base["vertices"],base["local"]):
                points=np.asarray(points)
                central=np.argsort(np.linalg.norm(points-np.median(points,axis=0),axis=1))[:5]
                selected.extend(np.asarray(ids)[central].tolist())
            tracked[arm]=np.unique(selected)
    texture_image=wrapper_texture(cracker=cracker)
    texture_image.save(args.run/"wrapper-print.png")
    if has_film:
        q0=data["q_00000"]
        uv=np.column_stack(((q0[:,1]-q0[:,1].min())/np.ptp(q0[:,1]),
                            (q0[:,0].max()-q0[:,0])/np.ptp(q0[:,0])))
    film_material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.,1.,1.,1.),
        baseColorTexture=pyrender.Texture(source=np.asarray(texture_image),source_channels="RGB"),
        metallicFactor=.15,roughnessFactor=.45,doubleSided=False)
    liner_material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.72,.75,.77,1.),
        metallicFactor=.45,roughnessFactor=.38,doubleSided=False)
    validated=False
    assessment=args.run/"core-assessment.json"
    if has_film and assessment.exists():
        report=json.loads(assessment.read_text())
        validated=(report.get("core_passed") is True and
                   report.get("states_sha256")==hashlib.sha256(recording_bytes).hexdigest() and
                   report.get("trace_sha256")==hashlib.sha256(trace_bytes).hexdigest())
    pinch_verified=False
    if pinch_test and (args.run/"pinch-lift-assessment.json").exists():
        report=json.loads((args.run/"pinch-lift-assessment.json").read_text())
        pinch_verified=(report.get("passed") is True and
            report.get("states_sha256")==hashlib.sha256(recording_bytes).hexdigest() and
            report.get("trace_sha256")==hashlib.sha256(trace_bytes).hexdigest())
    keys = sorted(k for k in data.files if k.startswith("q_" if has_film else "b_") and
                  args.first_step<=int(k[2:])<len(trace) and
                  (args.last_step is None or int(k[2:])<=args.last_step))[::args.stride]
    output = args.run / f"{'unfolded-yam' if has_film else 'rigid-clip'}-{args.detail}-{'core-demo' if validated else 'diagnostic'}.mp4"
    if pinch_test:output=args.run/f"pinch-lift-{args.detail}-{'pass' if pinch_verified else 'diagnostic'}.mp4"
    writer = imageio.get_writer(str(output), fps=hz/args.stride, codec="libx264",
                               quality=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    for key in keys:
        step = int(key[2:])
        for node in film_nodes:scene.remove_node(node)
        film_nodes=[]
        if has_film:
            film = trimesh.Trimesh(vertices=data[key], faces=data["faces"], process=False)
            film.visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=texture_image)
            film_nodes.append(scene.add(pyrender.Mesh.from_trimesh(film, smooth=False,material=film_material)))
            # A separate back-facing draw changes appearance only. The
            # physics surface remains the single frozen recorded mesh.
            liner=trimesh.Trimesh(vertices=data[key],faces=data["faces"][:,::-1],process=False)
            film_nodes.append(scene.add(pyrender.Mesh.from_trimesh(liner,smooth=False,material=liner_material)))
            if pinch_test and trace[step].get("pinch_lift",{}).get("reference_locked"):
                for arm,ids in tracked.items():
                    color=[.1,.9,.2] if arm=="left" else [.15,.4,1.]
                    film_nodes.append(scene.add(pyrender.Mesh.from_points(data[key][ids],colors=np.tile(color,(len(ids),1)))))
        bodies = data["b_"+key[2:]]
        for part, node in nodes:
            transform = np.array(part["local"])
            if part["body"] >= 0:
                transform = pose(bodies[part["body"]]) @ transform
            scene.set_pose(node, transform)
        views=[]
        detail_eye,detail_target=center+[-.10,-.03,.40],center+[-.065,0,0]
        if args.detail=="mouth_side":
            # Oblique side view of the mouth, away from the torso and elbow.
            detail_eye=center+[.05,-.28,.28]
        if args.detail=="mouth_front":
            detail_eye=center+[-.20,-.18,.17]
            detail_target=center+[-.075,0,.005]
        if args.detail=="clip":
            detail_target=np.asarray(trace[step]["clip_root_pose"][:3])
            detail_eye=detail_target+[.18,-.25,.20]
        for view_index,(eye, target) in enumerate(((center+[.75, -.60, .65], center+[-.10, 0, .08]),
                            (detail_eye,detail_target))):
            changed=[]
            if view_index==1 and args.translucent_robot_detail:
                for part,node in nodes:
                    if "yam_bimanual" not in part["name"]:continue
                    for primitive in node.mesh.primitives:
                        material=primitive.material
                        changed.append((material,material.baseColorFactor.copy(),material.alphaMode))
                        color=material.baseColorFactor.copy();color[3]=.16
                        material.baseColorFactor=color;material.alphaMode="BLEND"
            scene.set_pose(camera, camera_pose(eye, target))
            rgb, _ = renderer.render(scene)
            views.append(rgb)
            for material,color,mode in changed:
                material.baseColorFactor=color;material.alphaMode=mode
        canvas = np.full((720 if pinch_test else 640, 1280, 3), 245, np.uint8)
        canvas[55:595] = np.concatenate(views, axis=1)
        if args.translucent_robot_detail:
            cv2.putText(canvas,"Close-up: robot translucent; recorded geometry unchanged",(655,80),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(30,40,50),1,cv2.LINE_AA)
        if pinch_test:
            cv2.putText(canvas,"Green / blue dots: original left / right film grasp points",(655,102),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(30,40,50),1,cv2.LINE_AA)
        label = trace[step]["phase"]
        title="YAM + UNFOLDED FILM POUCH" if has_film else "YAM + SPRING CLIP | NO FILM"
        validation_label=(" | CONTENTS-PROTECTED MECHANICS" if cracker else " | EMPTY-POUCH MECHANICS")
        if pinch_test:
            title="YAM | TWO-HAND MOUTH GRASP + LIFT"
            validation_label=" | WITH CONTENTS" if arguments.get("contents") else " | EMPTY BAG"
        cv2.putText(canvas, title+(validation_label if validated else " | contact-driven diagnostic"), (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 40, 50), 1, cv2.LINE_AA)
        if pinch_test:
            cv2.rectangle(canvas,(0,0),(1280,52),(245,245,245),-1)
            cv2.putText(canvas,title+validation_label,(20,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(30,40,50),1,cv2.LINE_AA)
        scope=("3s hold PASS; contents retained, no seam failure" if cracker else "3s hold PASS; empty pouch") if validated else "DIAGNOSTIC; sealing not validated"
        if pinch_test:scope="MOUTH GRASP/LIFT 3s PASS" if pinch_verified else "MOUTH GRASP/LIFT NOT VERIFIED"
        cv2.putText(canvas, f"t={step/hz:.2f}s | {label} | recorded physics, 1x | {scope}",
                    (20, 625), cv2.FONT_HERSHEY_SIMPLEX, .60, (30, 40, 50), 1, cv2.LINE_AA)
        if pinch_test:
            measured=trace[step].get("pinch_lift",{})
            for number,arm in enumerate(("left","right")):
                hand=measured.get("hands",{}).get(arm,{})
                area=hand.get("wall_area_mm2",[0.,0.])
                contact="YES" if min(hand.get("opposed_load_N",[0.,0.]))>=.02 else "NO"
                lift=hand.get("wall_lift_m");slip=hand.get("wall_slip_p90_m")
                text=f"{arm.upper()}: wall capture {area[0]:.0f} / {area[1]:.0f} mm2 | opposed contacts {contact}"
                if lift is not None:
                    retained=hand.get("wall_landmarks_retained")
                    marks="UNMEASURED" if retained is None else ("IN" if all(retained) else "OUT")
                    hold=hand.get("wall_hold_slip_p90_m",[])
                    creep=f"{max(hold)*1000:.1f} mm" if len(hold) else "--"
                    gap=hand.get("wall_finger_distance_m")
                    if gap is not None:
                        text=f"{arm.upper()}: film rise {lift[0]*1000:.1f}/{lift[1]*1000:.1f} mm | finger gap {max(gap)*1000:.2f} mm | pad-edge {marks} (diag) | drift {creep} | contacts {contact}"
                    else:
                        text=f"{arm.upper()}: film rise {lift[0]*1000:.1f}/{lift[1]*1000:.1f} mm | original points {marks} | hold drift {creep} | total shift {max(slip)*1000:.1f} mm | contacts {contact}"
                cv2.putText(canvas,text,(20,651+number*25),cv2.FONT_HERSHEY_SIMPLEX,.52,(30,40,50),1,cv2.LINE_AA)
            legend=("NO FOLD / NO CLIP | contact + raised film judge retention; edge flags diagnostic | mouth lift only"
                    if measured.get("hands",{}).get("left",{}).get("wall_finger_distance_m") is not None else
                    "NO FOLD / NO CLIP | mouth lift only; whole-bag suspension is a separate check")
            cv2.putText(canvas,legend,(20,703),
                        cv2.FONT_HERSHEY_SIMPLEX,.52,(30,40,50),1,cv2.LINE_AA)
        writer.append_data(canvas)
        if args.stride>=30 or step in (0, 60, 300, 600, 900) or key == keys[-1]:
            imageio.imwrite(args.run/f"view-{step:05d}.png", canvas)
    writer.close()
    renderer.delete()
    (args.run/"render-manifest.json").write_text(json.dumps(dict(
        recording_sha256=hashlib.sha256(recording_bytes).hexdigest(),
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_steps=[int(key[2:]) for key in keys], fps=hz/args.stride,detail=args.detail,
        physics_modified=False,position_interpolation=False,
        translucent_robot_in_closeup=args.translucent_robot_detail,
        core_assessment_verified=validated,
        pinch_lift_assessment_verified=pinch_verified,
        material_vertex_dots=pinch_test,
        wrapper_print="original procedural outer print, fixed UV, silver inner liner; no collision effect"),indent=2))
    print(output)


if __name__ == "__main__":
    main()
