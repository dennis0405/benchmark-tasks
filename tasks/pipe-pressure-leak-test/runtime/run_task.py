"""Run one isolated task with scripted or budget-guarded Dream control and MP4 recording."""
import argparse
import hashlib
import html
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import time
import uuid
import sys

from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',required=True,help='Task module; all task-specific code stays alongside it')
p.add_argument('--seed',type=int,default=0)
p.add_argument('--max-steps',type=int,help='Optional local control-step cap; not a new benchmark rule')
p.add_argument('--output-dir',default='/outputs')
p.add_argument('--keep-open',action='store_true')
p.add_argument('--episodes',type=int,default=2)
p.add_argument('--policy',choices=('scripted','dream'),default='scripted')
p.add_argument('--budget-path',default='/dream-budget/ledger.sqlite3')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args(); args.enable_cameras=True
if args.policy=='dream' and args.episodes!=1:p.error('Dream runs exactly one episode')
print('[task] Starting Isaac Sim application',flush=True)
app=AppLauncher(args).app
print('[task] Isaac Sim application ready; loading task',flush=True)

import imageio.v3 as iio
import numpy as np
import torch
from yam_arena import station, ik
from yam_arena.constants import READY, FINGER_OPEN
from yam_arena.demo import ArmPlan
from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.scene import object_states

path=Path(args.task).resolve()
sys.path.insert(0,str(path.parent))
source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('selected_task',path)
task=importlib.util.module_from_spec(spec);sys.modules[spec.name]=task;spec.loader.exec_module(task)
object_names=tuple(getattr(task,'OBJECT_NAMES',getattr(task,'PARTS',{})))
if not object_names:raise ValueError('Task must declare OBJECT_NAMES')
manifest=json.loads(path.with_name('task.json').read_text())
if task.INSTRUCTION!=manifest['instruction']:raise ValueError('Task instruction must match task.json exactly')
if args.policy=='dream' and task.HZ!=20:raise ValueError('The pinned Dream adapter requires 20 Hz')
step_limit=min(task.MAX_STEPS,args.max_steps) if args.max_steps is not None else task.MAX_STEPS
if step_limit<1:raise ValueError('Step cap must be positive')
output_dir=Path(args.output_dir).resolve()
root=output_dir/task.NAME/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
root.mkdir(parents=True)
shutil.copytree(path.parent,root/'task-source',ignore=shutil.ignore_patterns('__pycache__','.runtime','.runs','.git','*.env','*.sqlite3*'))
shutil.copy2(__file__,root/'run.py')
for source in ('dream_runner.py','dream_budget.py'):
    shutil.copy2(Path(__file__).with_name(source),root/source)
source_snapshot=root/'task-source'
source_hashes={str(f.relative_to(source_snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
    for f in source_snapshot.rglob('*') if f.is_file()}
cameras=('overview_cam','top_cam','left_cam','right_cam')
emb=BimanualYamEmbodiment(enable_cameras=True,contact_sensors=False,rig=station.rig(),overview_cam=True)
emb.camera_config.overview_cam.width=1280;emb.camera_config.overview_cam.height=720
eye=np.array([.95,-.80,1.45]); target=np.array([.22,0,.80])
forward=(target-eye)/np.linalg.norm(target-eye)
left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
up=np.cross(forward,left)
emb.camera_config.overview_cam.offset.pos=tuple(eye)
emb.camera_config.overview_cam.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack([forward,left,up])))
env=None
writers={}
dream_runner=None
def frame(name):
    x=env.unwrapped.scene[name].data.output['rgb']
    x=x.torch if hasattr(x,'torch') else x
    return np.ascontiguousarray(x[0,:,:,:3].cpu().numpy())

def read_objects():
    # Arena's stock helper supports rigid objects only. Tasks with articulated
    # instruments can provide measured body poses and documented physical bounds.
    if hasattr(task,'read_object_states'):
        states,poses=task.read_object_states(env)
    else:
        states=object_states(env,object_names)
        poses={n:env.unwrapped.arena_world.get_pose_w(n)[0].cpu().numpy() for n in object_names}
    if set(states)!=set(object_names) or set(poses)!=set(object_names):
        raise ValueError('Measured object inventory differs from OBJECT_NAMES')
    return states,poses

def heartbeat(step,summary=None):
    data={'ready':True,'scene':task.NAME,'control_steps':step,'cameras':list(cameras),
          'updated':time.time(),'run_dir':str(root),'result':summary,'policy':args.policy}
    tmp=output_dir/'status.tmp';tmp.write_text(json.dumps(data));tmp.replace(output_dir/'status.json')

def finish_writers():
    errors=[]
    for name,(proc,log) in writers.items():
        proc.stdin.close()
        if proc.wait(timeout=30): errors.append(name)
        log.close()
    writers.clear()
    if errors: raise RuntimeError(f'Video encoding failed: {errors}')

def write_report(results):
    sections=[]
    labels={'overview_cam':'전체 장면','top_cam':'상단','left_cam':'왼쪽 손목','right_cam':'오른쪽 손목'}
    for r in results:
        directory=f"episode-{r['episode']:02d}"
        videos=''.join(f'<figure><figcaption>{label}</figcaption><video controls preload="metadata" '
                       f'poster="{directory}/{cam}-setup.png" src="{directory}/{cam}.mp4"></video></figure>'
                       for cam,label in labels.items())
        sections.append(f'<section><h2>{html.escape(r["policy"])} {r["episode"]} · {r["score"]:.1f} / 1.0</h2>'
                        f'<p>{r["steps"]} steps · 시뮬레이션 {r["sim_seconds"]:.2f}초 · '
                        f'Dream API {r["dream_api_calls"]}회 · 종료: {html.escape(r["stop_reason"])} · '
                        f'<a href="{directory}/result.json">측정 결과</a></p>'
                        f'<div class="videos">{videos}</div></section>')
    title=html.escape(manifest['title'])
    instruction=html.escape(task.INSTRUCTION)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Robot task recording</title><style>body{margin:32px auto;padding:0 20px;max-width:1200px;background:#111820;color:#eaf0f5;font:16px system-ui}
h1{font-size:28px}p{color:#bac8d6}a{color:#8ccaff}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}
figure{margin:0}video{width:100%;background:#000;border-radius:8px}figcaption{padding:8px 0}section{margin:36px 0}
@media(max-width:700px){.videos{grid-template-columns:1fr}}</style>'''+f'<h1>{title}</h1><p>{instruction}</p>'
    (root/'index.html').write_text(page+''.join(sections)+'</html>')

try:
    print('[task] Building physical environment',flush=True)
    env=task.make_environment(emb,args.device)
    print('[task] Physical environment ready',flush=True)
    dynamics=task.make_dynamics(env) if hasattr(task,'make_dynamics') else None
    ready=torch.tensor([[*READY,1,*READY,1]],dtype=torch.float32,device=env.unwrapped.device)
    results=[]
    for episode in range(args.episodes):
        out=root/f'episode-{episode+1:02d}';out.mkdir()
        snapshots=out/'snapshots';snapshots.mkdir()
        scorer=task.make_scorer()
        env.reset(seed=args.seed)
        print(f'[task] Reset episode {episode+1}; settling 30 frames',flush=True)
        if dynamics is not None:dynamics.reset()
        for settling_step in range(-30,0):
            if dynamics is not None:dynamics.before_step(settling_step,ready[0].cpu().numpy())
            env.step(ready)
        plans={side:ArmPlan(points,station.BASES[side]) for side,points in task.demo().items()} if args.policy=='scripted' else {}
        controller=task.make_demo_controller(env,plans) if args.policy=='scripted' and hasattr(task,'make_demo_controller') else None
        print('[task] Settling complete; opening camera recorders',flush=True)
        duration=max((x.duration for x in plans.values()),default=float('inf'))
        for name in cameras:
            rgb=frame(name);iio.imwrite(out/f'{name}-setup.png',rgb)
            height,width=rgb.shape[:2]
            log=(out/f'{name}-ffmpeg.log').open('w')
            cmd=['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{width}x{height}','-r',str(task.HZ),'-i','pipe:0','-an','-c:v','libx264',
                 '-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{name}.mp4')]
            writers[name]=(subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log),log)
        held=0; dropped=False; started=time.monotonic(); peak=0.0
        stop_reason='local_step_limit' if step_limit<task.MAX_STEPS else 'step_limit'
        steps=0; value=0.0; clear=True; m={}; clock=0.0; success=False
        action=np.array([*READY,1,*READY,1],dtype=np.float64)
        trace=(out/'trajectory.jsonl').open('w')
        initial_objects,_=read_objects()
        initial={n:{'pos':s.pos.tolist(),'quat_xyzw':s.quat_xyzw.tolist()} for n,s in initial_objects.items()}
        startup_error=None
        if args.policy=='dream':
            from dream_runner import Runner
            try:
                dream_runner=Runner(task.NAME,root.name+'-'+uuid.uuid4().hex[:12],task.INSTRUCTION,args.budget_path,seed=args.seed)
            except Exception as error:
                startup_error='dream_setup_error:'+type(error).__name__
                print('[dream] '+startup_error,flush=True)
        for step in range(step_limit):
            clock=(step+1)/task.HZ
            request_error=startup_error
            if args.policy=='scripted':
                action=controller.action(clock) if controller is not None else np.concatenate([plans[side].action7(clock)[0] if side in plans else np.r_[READY,1] for side in ('left','right')])
                action=np.asarray(action,dtype=np.float64)
                if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Demo action must be 14 finite joint/gripper commands')
            elif dream_runner is not None:
                state=[]
                for side in ('left','right'):
                    robot=env.unwrapped.scene[f'{side}_robot']
                    q=robot.data.joint_pos.torch[0].cpu().numpy(); names=robot.joint_names
                    state.extend(q[names.index(f'joint{i}')] for i in range(1,7))
                    state.append(float(np.clip(np.mean([q[names.index(f'joint{i}')]/FINGER_OPEN for i in (7,8)]),0,1)))
                try:
                    action=dream_runner.act({name:frame(name) for name in cameras if name!='overview_cam'},np.asarray(state))
                except Exception as error:
                    from dream_runner import RequestStopped
                    from dream_budget import BudgetStopped
                    request_error=str(error) if isinstance(error,(RequestStopped,BudgetStopped)) else 'dream_error:'+type(error).__name__
                    print('[dream] STOP '+request_error,flush=True)
            # Preserve a final recorded frame even if the first request/setup fails.
            if dynamics is not None:dynamics.before_step(step+1,action)
            env.step(torch.tensor(action,dtype=torch.float32,device=env.unwrapped.device)[None])
            steps=step+1
            objects,poses=read_objects()
            measured={}
            for side in ('left','right'):
                robot=env.unwrapped.scene[f'{side}_robot']
                q=robot.data.joint_pos.torch[0].cpu().numpy()
                names=robot.joint_names
                joints=np.array([q[names.index(f'joint{i}')] for i in range(1,7)])
                fingers=np.array([q[names.index(f'joint{i}')] for i in (7,8)])
                tcp,_=ik.tcp_pose(joints,station.BASES[side])
                measured[side]={'joints':joints.tolist(),'fingers':fingers.tolist(),'tcp':tcp.tolist()}
            mechanics=dynamics.after_step(step+1,objects,poses,measured) if dynamics is not None else None
            if mechanics is not None:
                # Task mechanics are scorer evidence only, never Dream observations.
                json.dumps(mechanics,allow_nan=False)
            m=scorer.update(step=step+1,objects=objects,root_poses=poses,robot=measured,
                            **({'mechanics':mechanics} if dynamics is not None else {}))
            clear=m.get('clear',True); dropped=m.get('dropped',False); held=m.get('held_frames',0)
            value=float(m['score'])
            success=bool(m.get('success',value>=1.0))
            if not 0<=value<=1:raise ValueError('Task score must be between zero and one')
            peak=max(peak,value)
            trace.write(json.dumps({'step':step+1,'sim_time_s':clock,'action':action.tolist(),
                                   'robot':measured,'objects':{n:{'pos':s.pos.tolist(),'quat_xyzw':s.quat_xyzw.tolist(),
                                   'lin_vel':s.lin_vel.tolist(),'ang_vel':s.ang_vel.tolist(),
                                   'low':s.low.tolist(),'high':s.high.tolist()} for n,s in objects.items()},
                                   'root_poses':{n:x.tolist() for n,x in poses.items()},
                                   **({'mechanics':mechanics} if dynamics is not None else {}),
                                   'metrics':m,'clear':clear,'dropped':dropped,'score':value})+'\n')
            for name,(proc,_) in writers.items():proc.stdin.write(frame(name).tobytes())
            if step==0 or (step+1)%40==0 or success or request_error or step+1==step_limit:
                # Camera-only review images: preserve history and atomically update latest.
                for name in ('overview_cam','top_cam'):
                    rgb=frame(name)
                    iio.imwrite(snapshots/f'{name}-step-{step+1:05d}.png',rgb)
                    temporary=out/f'{name}-latest.tmp.png'
                    iio.imwrite(temporary,rgb)
                    temporary.replace(out/f'{name}-latest.png')
            if step%40==0:
                heartbeat(step+1)
                print(f'[task] episode={episode+1} step={step+1} score={value} success={success}',flush=True)
            if request_error:
                stop_reason=request_error;break
            if success:
                stop_reason='success';break
            if clock>duration+5:
                stop_reason='script_finished';break
        trace.close();finish_writers()
        for name in cameras:iio.imwrite(out/f'{name}-final.png',frame(name))
        dream_report=None
        if dream_runner is not None:
            dream_runner.close()
            dream_report=dream_runner.report()
            (out/'dream.json').write_text(json.dumps(dream_report,indent=2))
            dream_runner=None
        api_calls=dream_report['budget']['reserved']['run_calls'] if dream_report else 0
        result={'task':task.NAME,'episode':episode+1,'success':success,
                'score':value,
                'peak_score':peak,'steps':steps,'sim_seconds':clock,'wall_seconds':time.monotonic()-started,
                'clear':clear,'dropped':dropped,'final_metrics':m,'initial':initial,
                'max_ik_mm':{s:p.worst_ik_mm for s,p in plans.items()},'dream_api_calls':api_calls,
                'policy':args.policy,'stop_reason':stop_reason,'dream':dream_report,
                'evaluation_complete':stop_reason in ('success','step_limit'),
                'instruction':task.INSTRUCTION,'seed':args.seed,'run_limit_steps':task.MAX_STEPS,
                'source_sha256':source_hashes,
                'source_manifest_version':2,
                'task_sha256':source_hash,'isaac_sim':'6.1.0','control_hz':task.HZ}
        (out/'result.json').write_text(json.dumps(result,indent=2));results.append(result)
        write_report(results)
        print('[task] RESULT '+json.dumps(result),flush=True)
    (root/'summary.json').write_text(json.dumps(results,indent=2))
    # Keep the final scene available to the existing WebRTC browser.
    while args.keep_open and app.is_running():
        env.step(torch.tensor(action,dtype=torch.float32,device=env.unwrapped.device)[None])
        heartbeat(step+1,results)
        time.sleep(.05)
except Exception:
    # App shutdown may mask the original traceback in some Kit builds.
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if dream_runner is not None:dream_runner.close()
    if writers:finish_writers()
    if env is not None:env.close()
    app.close()
