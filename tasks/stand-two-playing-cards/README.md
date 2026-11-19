# Stand two playing cards by leaning their top edges together

The robot builds the first pair of a card house. Each card starts in its own holder. After lifting and inclining the cards, the robot transfers their weight to their upper-edge contact and to the mat. Either hand may be released first.

## Run one trial

Use Linux with Docker, the NVIDIA Container Toolkit and an RTX-capable GPU. From this task directory, build the pinned Isaac Sim 6.1.0, Isaac Lab Arena and YAM environment with `python3 runtime/build_image.py`. The following command uses the default image tag and a local output directory.

```sh
mkdir -p .runs
docker run --rm --gpus all --shm-size=8g --user "$(id -u):$(id -g)" \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/tmp \
  -v "$PWD:/task:ro" -v "$PWD/.runs:/outputs:rw" \
  yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --seed 0 \
  --episodes 1 --max-steps 800 --device cuda:0 --livestream 0 --viz none
```

For the model policy, select `--policy dream`. The recorded run uses dreamlabs-client 1.0.14 and the pinned Inspect Robots YAM adapter. The model receives the instruction, overhead and wrist RGB views, and measured joint and gripper state. The overview camera records the trial.

A Dream evaluation requires the operator's API key in an owner-readable file and an episode reservation from the operator's budget ledger. Mount the credential file read-only at `/run/secrets/dream.env`, set `DREAM_CREDENTIALS=/run/secrets/dream.env`, mount the private reservation directory read/write at `/dream-budget`, and pass `--budget-path /dream-budget/account-reservation.json`. The file holding the key uses `DREAM_API_KEY=value`. The included budget adapter reads the task ID, account run ID and authorized cumulative reservation from that receipt and journals each transport request in the same directory. The evaluated episode used a USD 20 reservation. Reservations are spending estimates.

Each episode saves camera MP4s, a setup image, action and state trajectories, and result.json. In the same software environment, run the following command to check an episode's terminal metrics.

```sh
python runtime/rescore.py --task task.py /path/to/episode-01/trajectory.jsonl \
  --expect /path/to/episode-01/result.json
```

## Physical setup and reset

Use two stiff laminated jumbo playing cards, 89 mm wide, 127 mm tall and 0.65 mm thick, modeled at 8 g each. Place a 290 × 440 mm flat rubber mat centrally between the arms, with its long axis spanning the left and right sides. Two shallow holders sit 165 mm to either side of the mat center along that axis. Each slot is formed by two 5 × 77 × 20 mm rails separated by a 2 mm gap. Stand one card in each slot with its 89 mm edge down and parallel to the rails. Start with the red three on the left arm's side and the blue five on the right arm's side. Keep the central work area clear. Reset by returning the cards to the slots and the arms to their ready poses. Use seed 0 to reproduce the recorded layout.

In simulation coordinates, the mat's upper surface is at z = 0.775 m. The card centers start at x = 0.300 m and y = ±0.165 m. Each card is a separate free rigid body. The cards and mat have assumed static friction coefficients of 0.75 and 0.80, with dynamic coefficients 90% of those values. Printed pips are visual details. Card bending, rounded edges and surface wear are approximated. Physical material properties have not been measured.

## Scoring one episode

- Give 0.5 when both cards have their bottom edges on the mat and lean together along their upper edges for 0.50 continuous seconds, clear of the holders and other props. The robot can support the cards at this stage.
- Give 1.0 after the pair stands still and unsupported for 3.00 continuous seconds. Both arms and grippers must be out of contact. A hand may remain near the cards.
- Otherwise give 0. The score follows the current state. An interrupted interval starts over. A side-view recording and a timer show the release and hold in a real trial.

The native scorer checks contact pairs, top-edge overlap greater than 20 mm, top-edge separation below 6 mm, and bottom-edge height error below 2.5 mm. The observer checks opposing inward lean and excludes support from fixtures. Stillness is measured with center speeds below 12 mm/s and angular speeds below 0.16 rad/s. At 20 Hz the partial and full intervals require 11 and 61 consecutive samples. Only arm joints and grippers are actuated. The full run limit is 800 actions, or 40 simulated seconds.

## Recorded results

Both trials used the same task implementation, source assets, instruction, seed 0 and 800-action limit on Isaac Sim 6.1.0.

| Policy | Terminal / peak score | Actions | Simulated time | Wall time after setup |
| --- | --- | --- | --- | --- |
| Dream-1 | 0.0 / 0.0 | 800 | 40.00 s | 152.4 s |
| Scripted YAM | 1.0 / 1.0 | 558 | 27.90 s | 93.8 s |

Dream-1 reached the action limit. No robot-to-card or card-to-card contact was reported during its trial. The scripted policy first earned half credit at 25.20 s. The two cards then met the full three-second independent-hold criterion at 27.90 s.
