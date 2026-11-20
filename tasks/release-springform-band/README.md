# Release a cheesecake from its springform band

Remove the metal band from the cooled cheesecake and set it on the empty mat. Leave the cake upright and undisturbed on its original base.

## Run one trial

Use Linux with Docker, the NVIDIA Container Toolkit and an RTX-capable GPU. From this task directory, build the pinned Isaac Sim 6.1.0, Isaac Lab Arena and YAM environment with `python3 runtime/build_image.py`. The following command uses the default image tag and a local output directory.

```sh
mkdir -p .runs
docker run --rm --gpus all --shm-size=8g --user "$(id -u):$(id -g)" \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/tmp -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/task:ro" -v "$PWD/.runs:/outputs:rw" \
  yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --seed 0 \
  --episodes 1 --max-steps 860 --device cuda:0 --livestream 0 --viz none
```

For the model policy, select `--policy dream`. The model runner uses dreamlabs-client 1.0.14 and the pinned Inspect Robots YAM adapter. The model receives the instruction, overhead and wrist RGB views, and measured joint and gripper state. The overview camera and a fixed level side camera record the trial. The model receives the overhead and wrist views only.

A Dream evaluation requires the operator's API key in an owner-readable file and an episode reservation from the operator's budget ledger. Mount the credential file read-only at `/run/secrets/dream.env`, set `DREAM_CREDENTIALS=/run/secrets/dream.env`, mount the private reservation directory read/write at `/dream-budget`, and pass `--budget-path /dream-budget/account-reservation.json`. The file holding the key uses `DREAM_API_KEY=value`. The included budget adapter reads the task ID, account run ID and authorized cumulative reservation from that receipt and journals each transport request in the same directory. The model runner expects a USD 20 episode reservation. Reservations are spending estimates.

Each episode saves camera MP4s, a setup image, action and state trajectories, and result.json. In the same software environment, run the following command to check an episode's terminal metrics.

```sh
python runtime/rescore.py --task task.py /path/to/episode-01/trajectory.jsonl \
  --expect /path/to/episode-01/result.json
```

## Reset and physical setup

Place a cooled, edge-loosened cheesecake upright on the original base plate. Center the closed band around it with its rolled lower lip engaged under the plate rim. Orient the latch toward the robot. Place the empty destination mat beside the pan, with its center 194 mm from the starting center. Reset both arms and use seed 0.

The simulated cake measures 146 mm across and 48 mm high and weighs 480 g. The 150 mm plate and raised foot weigh 115 g together. Four linked sectors represent expansion of the 76 mm high metal band. During the recorded lift, the left gripper pinches the upper wall of the band and the right gripper holds the flat latch lever. The cake is a rigid approximation of a cooled cake with pre-loosened edges. The scorer measures cake and base motion and monitors contact on the exposed cake. Both cake layers move as one rigid body.

### Preparing a physical trial

1. Stage one intact, plain John & Jill’s New York **Retail 28 oz** cake per trial, preferably from the same production lot. The manufacturer lists a nominal 6 in diameter and 2 in height. Reject a pre-sliced or visibly damaged unit. If supplied frozen, thaw as directed on the package. Keep the staged samples covered and chilled. [Manufacturer’s size list](https://www.johnandjillscheesecake.com/new-york/)
2. Before the session, use a clean 146 mm round food cutter and a cake wire on 48 mm guide rails to trim a centered cylinder with its bottom intact. Measure two perpendicular diameters and height at four points. Accept 146 ± 1 mm across and 48 ± 1 mm high. Record lot, measured dimensions and mass. Check for a core temperature of 4 ± 1 °C on a companion unit stored alongside the samples. Take scaled photos of the top and four sides before fitting the band. Reject samples with a dent, gouge, split or missing piece. Sample mass, texture and trim yield still require physical verification.
3. For each reset, use a broad cake lifter to center a fresh prepared sample on the clean original plate. Seat the lower lip, close the latch and point it toward the arms. With a thin blunt palette knife against the inner metal wall, sweep vertically down to the base and once around the entire circumference. Confirm a second full pass without dragging the cake. Remove the knife and verify the rim remains seated. The task begins with the edges already loose. [Edge-loosening technique](https://www.kingarthurbaking.com/blog/2016/05/04/cheesecake-tips)
4. Fit clean, removable food-safe pad covers on both grippers before each trial. Keep trimming and photography outside the scored interval. Between trials, exchange the prepared sample, clean the plate and band, and replace or clean the pad covers. Record batch yield and reset time when commissioning the physical setup.

The [Fat Daddio’s PSF-63](https://shop.fatdaddios.com/products/springform-pans) is a reference for the springform mechanism. Its catalog designation is 6 × 3 inches. The real pan has a rolled upper rim and a flat latch lever. The simulation uses a 2 × 50 × 19 mm flat lever with assumed dimensions and four linked band sectors. The latch opens the band through a simplified mechanical linkage. Physical gripper compatibility with the commercial pan and the mass/size match must be checked before claiming equivalent performance.

## Scoring

- If the cake or base moves more than 3 mm from its initial center or tilts more than 3 degrees from upright at any time, the episode earns zero, including after it is returned to its original pose.
- Cake damage at any time also gives zero, even after the band clears the cake. In simulation, use the greatest summed normal impulse of any single exposed-cake contact manifold and the deepest contact point in each control interval. An impulse ≥ 0.025 N·s or penetration ≥ 2 mm latches damage for the episode. Include grippers and moving metal, and exclude contact with the supporting base. The impulse cutoff is a provisional proxy for hard pressing or impact. The rigid cake cannot develop a visible indentation or fracture. No physical yield threshold has been measured.
- Earn 0.5 when the lower lip clears the plate rim with adequate radial clearance around all 40 facets at the time of crossing, then rises above the cake. Lowering the band back around the original base cancels this credit.
- Earn 1.0 when the separated band rests on the destination mat with its center over the mat, free of contact with either arm and stable for one continuous second. Linear and angular speed must stay below 15 mm/s and 0.15 rad/s. A hand that still supports the band limits the score to 0.5.

For physical scoring, record synchronized close top and level side views of both mats and grippers. Put millimetre scales in the base and cake planes. Include fixed orientation references and visible marks on the base. Verify scale and angle with known displacements and a tilt reference before scoring. Read the lowest visible band edge relative to the highest cake point, check that the cake remains on its original base, and time a full second of visible clearance from both hands after placement. Record an occluded or borderline event as indeterminate. In the simulator, the level camera faces the pan between two fixed scales marked every 1 mm with longer ticks at 5 and 10 mm. A reviewer matched the scorer on one normal lift and one brief base lift with combined translation and tilt. These two videos did not test perception of translation and tilt independently near their cutoffs.

### Inspecting cake damage in a physical trial

Use the synchronized footage to catch damage during manipulation. Before lifting or touching the cake after the trial, repeat the scaled photographs at the original viewpoints, moving the camera around the cake. Compare the same features against the pre-trial photos. Any of the following gives zero, regardless of the final pose or band placement:

- A new dent, gouge or height loss of at least 2 mm on the top or side. Use a straightedge spanning adjacent intact surface and a millimetre scale in profile. For overall flattening, compare height above the original base.
- A new split with an opening at least 1 mm wide over at least 5 mm, or a detached piece of cake measuring at least 3 mm across its longest dimension.

Use synchronized video for transient crushing that disappears on release. If the relevant surface is obscured or the measurement is inconclusive, mark that trial indeterminate and repeat it with a fresh sample. Apply the same inspection before awarding either positive score. To calibrate the simulation proxy, compare intact controls and deliberately pressed or scraped spare samples at the same temperature and measure the resulting damage.

## Recorded trials

All trials used seed 0 and 20 Hz control. The normal demo, vertical press and oblique scrape ran in Isaac Sim with the same damage-aware scorer, pan and robot. The controls used only robot joint and gripper commands. After touching the cake, the robot continued with the usual release sequence.

| Revised scorer | Final score | Steps | Time | First damage | Peak normal impulse | Deepest contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted YAM, normal release | 1.0 | 603 | 30.15 s | None | 0 N·s | 0 mm |
| Press cake top, then release band | 0.0 | 860 | 43.00 s | 3.40 s | 0.17194 N·s | 0.00340 mm |
| Press obliquely and scrape, then release band | 0.0 | 860 | 43.00 s | 2.95 s | 0.10455 N·s | 0.00361 mm |

The normal demo cleared the cake at 18.10 s. Maximum cake or plate translation was 0.291 mm and tilt was 0.00113°. In the two contact controls, the band cleared at 27.15 s. Both remained within the pose limits and both earned 1.0 when independently rescored with only the damage check disabled. With damage scoring enabled, both triggered the impulse cutoff. Neither reached the 2 mm penetration cutoff. During contact, the right-hand TCP spanned 3.4 mm along the horizontal x axis in the press and 53.1 mm in the scrape.

### Dream-1 evidence and board media

The board model video is the original seed-0 Dream-1 evaluation. It completed 860 steps (43.00 s) with 19 successful service requests and scored 0.0. We replayed all 860 recorded 14-dimensional actions from a fresh reset with the revised contact observer. Every action matched the original log. At every logged step, the measured cake, plate and band centers matched the original run. The replay never cleared the band or triggered damage, and scored 0.0. Replay made zero model calls. The board demo and setup image come from the revised 1.0-point YAM run.

Timed execution on one Blackwell GPU, excluding simulator startup, took 105 s for the revised normal demo, 158 s for the press, 161 s for the scrape and 160 s for Dream action replay. The original online Dream evaluation took 173 s.

### Replaying the validation actions

The three compressed files under `validation/` contain the recorded 14 joint/gripper commands per step. To reproduce a control, run the following from the task directory, choose `press`, `gouge` or `dream`, and then run the scripted Docker command above from the temporary copy. Each validation replay starts from a fresh seed-0 reset. The ordinary scripted command in the original directory runs the normal YAM demo.

```sh
case_name=press
replay_dir=$(mktemp -d)
cp -R . "$replay_dir/task"
cd "$replay_dir/task"
cp validation/replay_controller.py controller.py
cp "validation/$case_name-actions.json.gz" actions.json.gz
```
