# Sweep scattered crumbs into a handheld dustpan and lift them clear

One hand steadies the pan while the other sweeps two patches of debris. The controller then withdraws the brush and lifts the loaded pan. Actors and scoring are local to this directory. The scripted controller reads tool poses and sends only the 14 arm and gripper actions. Dream receives the instruction, calibrated top and wrist RGB cameras, and measured joint and gripper state.

## Scene and scoring

The pan lip starts at world (300, −50, 748) mm and the brush at (262, −255, 748) mm. The surface is at 747.5 mm. The eight freely simulated pieces weigh 0.8 g each. Seed 0 restores the same starting poses. The equipped YAM station models small-area sweeping within reach. The scene uses two original rigid compound tools, a convex bevel and two crumb sizes.

At 20 Hz the scorer tracks each piece separately. A forward crossing must occur inside the mouth, with both lip ends within 3 mm of the surface. Each tool grip must be within 35 mm of a distinct hand TCP with finger separation of 6–45 mm. The bristle volume must meet the moving piece within 2.5 mm, or a moving chain of touching pieces within 1.5 mm must connect them. The brush and credited pieces must each advance at least 0.1 mm toward the pan in that observation. Piece and tool bounding boxes are geometric contact proxies. A piece within 32 mm of either hand TCP is conservatively excluded for the episode. These tests approximate physical contact and may reject a near-hand pass or a piece that coasts in after the brush stops.

A piece must stay within the pan after entry. The current score pays 1/16 per valid piece wholly inside the deck bounds, supported within 3 mm of the 6 mm deck, with the pan facing upward. It pays another 1/16 after 61 consecutive observations (60 intervals, 3 seconds) with that piece inside and clear of the bristles, a hand on the grip and all four lowest deck corners at least 50 mm above the table. Loss of height, support or grip resets the affected hold. Leaving the mouth or the tray envelope requires fresh swept entry. End-of-episode score is reported at success or the step limit.

For a physical trial, put a 190 mm wide handheld dustpan and 93 mm hand brush on a flat surface and place eight rigid crumbs at the offsets in task.py. Record the lip and all pieces from above and the side with a scale at surface level. Count pieces that cross the lip under brushing. Verify the least of the four base-corner clearances and continuous containment for three seconds. The supplied USDs specify the geometry. Friction, gripper and bristle response have not been calibrated against hardware.

## Reproduction

The pinned image uses Isaac Sim 6.1.0, a CPU PhysX solver at 240 Hz and a Blackwell GPU for RTX rendering. The model has up to 720 actions (36 simulated seconds) at 20 Hz. Start from this task directory with the portable runtime. Build the image with `python3 runtime/build_image.py` if needed.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" --group-add 1234 -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```

For Dream use `--policy dream` with the same scene, seed and limit. Mount the operator's credentials file at /run/secrets/dream.env, set DREAM_CREDENTIALS to that path, and mount the operator's authorized portable budget directory at /dream-budget. Run `python runtime/rescore.py --task ./task.py /path/to/episode/trajectory.jsonl --expect /path/to/episode/result.json` in the pinned environment to compare the live and replayed scores.

## Context and prior work

Sweeping floors is a listed duty for US janitors and building cleaners. [O*NET occupation](https://www.onetonline.org/link/summary/37-2011.00) · [BLS Occupational Outlook Handbook](https://www.bls.gov/ooh/building-and-grounds-cleaning/janitors-and-building-cleaners.htm)

RoCoBench Sweep already coordinates a broom and movable dustpan and transfers cubes to a bin. Here the measured outcome is the simultaneous retention of eight scattered pieces under a specified lift and dwell. BEHAVIOR-1K also includes sweeping floors. [RoCoBench appendix](https://project-roco.github.io/roco-icra-appendix.pdf) · [BEHAVIOR-1K task](https://behavior.stanford.edu/knowledgebase/tasks/sweeping_floors-0.html)
