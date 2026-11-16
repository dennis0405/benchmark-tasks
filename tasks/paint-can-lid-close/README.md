# paint-can-lid-close

**Instruction:** "Close the paint can."

A half-full 1-gallon paint can stands on a canvas drop cloth. Its lid has been set on the ring but not
tapped down: it sits about 4 mm proud, with wet paint in the ring's groove. An 8 oz rubber mallet lies
on the cloth beside it. The lid is closed by tapping its rim all the way round with the mallet while the
other hand steadies the can.

## Why this job

Closing a part-used can so the paint does not skin over is routine for painters (about 370,000 in the
US), general maintenance and repair workers (about 1.5 million) and construction laborers, and a common
household job. A ring-and-plug lid is set on square and tapped down with a rubber mallet round its whole
rim until every part of it is seated flush in the can's groove, while the other hand steadies the can
on its drop cloth.

## Real parts (all real size)

| part | real item | size / mass |
|---|---|---|
| paint can | 1 US gallon (3.78 L) open-top steel can, ring-and-plug lid, wire bail with plastic grip, printed paper label (made-up maker) | 168 mm across the chimes, 190 mm tall; about half full, 2.4 kg |
| lid | tinplate plug lid, tint-recipe sticker | 157 mm across the curl, 8 mm plug, 70 g |
| mallet | 8 oz rubber mallet, wooden handle | head 51 x 92 mm, handle 25 mm, 305 mm overall, 0.31 kg |
| drop cloth | folded cotton canvas, old paint spots | 620 x 500 x 4 mm |

The label and sticker art (`textures/`) is made by `make_textures.py` from free fonts (Liberation: SIL OFL
1.1; DejaVu: Bitstream Vera licence); the brand is invented.

## Scoring

The lid's rim is 8 sectors of 45 degrees round the can's axis. A sector is seated by a tap: the mallet's
lower face coming down on it (face centre 40-95 mm from the axis, within 4 mm of the lid's top, moving down
at 0.04 m/s or more, face within 25 degrees of flat).

| score | condition |
|---|---|
| 0.25 | at least 3 sectors seated |
| 0.5 | at least 6 sectors seated |
| 0.75 | all 8 seated and the lid down within 1 mm of seated (flush in the ring) |
| 1.0 | also the mallet laid back on the cloth at rest (head centre under 60 mm up, under 0.02 m/s and 0.2 rad/s) and every finger pad at least 50 mm from the mallet, the can and the lid, held 1 s |

- The can pushed more than 15 mm or tipped more than 8 degrees at any time caps the score at 0.5.
- The mallet off the bench, or the can knocked over, scores 0.
- Stages latch once reached. The finger pads are oriented boxes placed from the measured joints and jaw
  opening by the URDF's forward kinematics (`task.hand_points`, at `station.BASES`).

## Simulation notes

- PhysX 240 Hz. Every part has a detailed visual mesh over simple hidden colliders; the label and sticker
  are printed images of a made-up maker.
- The paint is not simulated: a static wet surface in the groove and drips down the wall.
- The can is one rigid body carrying its contents' mass (2.4 kg), centre of mass low; it stands free on the
  cloth.
- The plug lid is a vertical slide on the can (its plug in the ring's groove holds it sideways), stopped
  4 mm proud of seated.
- The seal is a stated rule (`task.Dynamics`): taps seat sectors as above, from measured poses only. Once
  all 8 are seated the slide's stop is lowered 4 mm and the lid falls flush under its weight and the taps.
  No position targets; nothing snaps. The lid and can do not dent.

## Scripted demo

1. The right hand takes the mallet's handle from above right by the head (at its centre of mass, so the
   head-heavy mallet does not swing in the pinch), lifts it and turns it a quarter turn about the handle so
   the head's faces point up and down.
2. The left hand braces the can's wall from the +y side, low down and clear of the strikes.
3. The right hand lifts the mallet up over the can and taps the lid's rim sector by sector round the can
   (180, 225, ... 135 degrees), the handle heading back toward the right arm.
4. With all 8 sectors seated the lid drops flush. The left hand lets go; the right hand turns the mallet
   flat again, lays it back on the cloth and both hands return to rest.

Recorded: 1.0 at step 729 (36.5 s simulated) of the 1,800-step limit; the can moved 0.1 mm.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the `yam-inspect-isaac:6.1.0`
image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`), write outputs outside the task
folder (the runner snapshots the task folder, so an output folder inside it would be copied into itself):

```bash
OUT="$(cd .. && pwd)/paint-can-lid-close-outputs"
mkdir -p /tmp/rc-task-home/tmp "$OUT"
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$OUT":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python runtime/rescore.py --task task.py "$OUT"/TASK/RUN/episode-01/trajectory.jsonl \
  --expect "$OUT"/TASK/RUN/episode-01/result.json
```

Seed 0; fixed poses, no randomisation; the scene settles for 30 ticks after reset before control starts.
Control 20 Hz, physics 240 Hz. A trial takes about 36 s simulated (about 2-4 minutes of wall time).

## Files

`paint.py` (geometry, parts, visual meshes), `task.py` (scene and demo), `scoring.py`, `make_textures.py`
and `textures/` (printed art), `simkit.py` and `meshlib.py` (parts, joints, meshes, motion helpers),
`run.py`.
