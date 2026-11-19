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

For the model policy, select `--policy dream`. The model runner uses dreamlabs-client 1.0.14 and the pinned Inspect Robots YAM adapter. The model receives the instruction, overhead and wrist RGB views, and measured joint and gripper state. The overview camera records the trial.

A Dream evaluation requires the operator's API key in an owner-readable file and an episode reservation from the operator's budget ledger. Mount the credential file read-only at `/run/secrets/dream.env`, set `DREAM_CREDENTIALS=/run/secrets/dream.env`, mount the private reservation directory read/write at `/dream-budget`, and pass `--budget-path /dream-budget/account-reservation.json`. The file holding the key uses `DREAM_API_KEY=value`. The included budget adapter reads the task ID, account run ID and authorized cumulative reservation from that receipt and journals each transport request in the same directory. The model runner expects a USD 20 episode reservation. Reservations are spending estimates.

Each episode saves camera MP4s, a setup image, action and state trajectories, and result.json. In the same software environment, run the following command to check an episode's terminal metrics.

```sh
python runtime/rescore.py --task task.py /path/to/episode-01/trajectory.jsonl \
  --expect /path/to/episode-01/result.json
```

## Reset and physical setup

Place a cooled, edge-loosened cheesecake upright on the original base plate. Center the closed band around it with its rolled lower lip engaged under the plate rim. Orient the latch toward the robot. Place the empty destination mat beside the pan, with its center 194 mm from the starting center. Reset both arms and use seed 0.

The simulated cake measures 146 mm across and 48 mm high and weighs 480 g. The 150 mm plate and raised foot weigh 115 g together. Four linked sectors represent expansion of the 60 mm high metal band. During the recorded lift, the left gripper pinches a raised flange on one lift ear and the right gripper keeps hold of the opened lever by its upright cylindrical tip. The cake is a rigid approximation of a cooled cake with pre-loosened edges. The task measures displacement and tilt; it does not model adhesion, tearing or indentation.

### Preparing a physical trial

Use the same batch recipe for all trials: [Dessert for Two’s plain 6-inch baked cheesecake](https://www.dessertfortwo.com/6-inch-cheesecake-recipe/), including its graham crust and water-bath bake. Omit toppings. Use the original removable base, mark its orientation and tare the assembled empty pan before filling. Chill the cooled cake in its latched pan overnight. Start trials at a center temperature of 4 ± 1 °C, measured with a thin food probe; withdraw the probe before reset. Record the actual chilled cake mass (assembled weight minus tare), height and diameter. The simulation’s 480 g, 48 mm and 146 mm are model values. A matched yield from this recipe has yet to be measured.

Pass a thin blunt palette knife vertically between the cold cake and the inner wall, down to the base, for one complete turn. Keep the blade against the wall. Confirm a second full pass to the base without sticking or pulling the cake. Remove the blade. Verify the cake is upright, the base is seated, the latch is closed and all tools and temporary markers used for preparation are clear of the grasp path. Reject a cracked, displaced or still-adhering sample. Use a freshly prepared sample for comparisons after a disturbed trial. The scored task starts after this pre-release check. Surface damage is outside the simulated score. [Edge-loosening technique](https://www.kingarthurbaking.com/blog/2016/05/04/cheesecake-tips)

The [Fat Daddio’s PSF-63](https://shop.fatdaddios.com/products/springform-pans) is a reference for the springform mechanism. Its catalog designation is 6 × 3 inches. The real pan has a rolled upper rim and a flat latch lever. The recorded fixture has two added grip ears, a cylindrical lever tip and four linked band sectors. The recorded scores apply to that fixture. Physical gripper compatibility with the commercial pan and the mass/size match must be checked before claiming equivalent performance.

## Scoring

- If the cake or base moves more than 2 mm or tilts more than 2 degrees at any time, the episode earns zero, including after it is returned to its original pose.
- Earn 0.5 when the lower lip clears the plate rim with adequate radial clearance around all 40 facets at the time of crossing, then rises above the cake. Lowering the band back around the original base cancels this credit.
- Earn 1.0 when the separated band rests on the destination mat with its center over the mat, free of contact with either arm and stable for one continuous second. Linear and angular speed must stay below 15 mm/s and 0.15 rad/s. A hand that still supports the band limits the score to 0.5.

For physical scoring, record synchronized close top and level side views of both mats and grippers. Put millimetre scales in the base and cake planes. Include fixed orientation references and visible marks on the base. Verify scale and angle with known displacements and a tilt reference before scoring. Read the lowest visible band edge relative to the highest cake point, check that the cake remains on its original base, and time a full second of visible clearance from both hands after placement. Record an occluded or borderline event as indeterminate. The original wide cameras do not resolve every 2 mm or 2° violation. Per-facet physical clearances and exact release-contact frames cannot be inferred from those views.

## Scripted demonstration

YAM opens the latch, lifts the band clear of the cake at 17.85 s, then sets it on the adjacent mat. Both arms are clear during the final one-second hold. The native scorer awarded 1.0 at step 640 (32.00 s). The maximum measured displacement of the cake or base was 1.15 mm.

## Recorded episodes

Both policies used the same scene, scorer, instruction, seed 0 and 860-step limit on Isaac Sim 6.1.0. The recordings follow their original 20 Hz action sequences.

| Policy | Final score | Actions | Simulated time | Episode wall time after setup |
| --- | ---: | ---: | ---: | ---: |
| Dream-1 | 0.0 | 860 | 43.00 s | 151.0 s |
| Scripted YAM | 1.0 | 640 | 32.00 s | 95.9 s |

Dream-1 ran to the step limit. The contact observer recorded no arm contact with the latch or band and the lower lip stayed on the starting side of the plate rim. Replaying both state traces with the submitted scorer reproduces every per-step score.
