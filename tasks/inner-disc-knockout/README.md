# Open only the smaller conduit entry

A YAM drives a fixed flat blade into the center of the concentric knockout on an empty electrical enclosure. The opposite arm stays clear. Pulling out the larger ring spoils the selected entry diameter.

## Set up the scene

Start with a fresh enclosure, its cover removed, its back secured to a bench block and its target side facing the tool. The commercial size reference is the nVent HOFFMAN ASE4X4X3. The simulated slug is 28.3 mm in diameter, the larger knockout is 34.6 mm across and the sheet is 1.52 mm thick. The four walls and bottom obstruct the original rear approach. The blade is 5.5 mm wide and 1.2 mm thick at its tip. The adjacent entries are visual stamp outlines. See [physical reference, reset and gauge protocol](REAL_TRIAL.md) for source drawings, approximations and occupational evidence.

## Occupational evidence

BLS reports 821,000 US electrician jobs in 2025. O*NET lists conduit and wiring installation as core electrical work. The number of US workers who regularly remove this specific knockout is unverified, including whether it reaches 500,000. See the [sources and physical protocol](REAL_TRIAL.md).

## Score at the end of the 12-second trial

| Score | What the observer should verify |
|---|---|
| 1.0 | The center disk separates, the entire smaller aperture is clear and the outer ring is retained. Both hands and the blade stay clear of the work for the final second. A quick clean release can earn full credit between control frames. |
| 0.5 | The disk remains attached and visibly bent. Its tilt exceeds 14 degrees, the two ligament marks remain within 4 mm and the two outlines still overlap. |
| 0.0 | The ring loses retention, shifts over 2 mm or tilts over 8 degrees at any point. A detached slug still covering the opening, an untouched entry or an insufficient final hold also scores zero. |

During simulation the scorer projects the full thickness and rim of the disk onto the plane of the opening. Full credit requires a separation margin over 2 mm and 20 consecutive contact-free control samples with both tool centers and the blade tip over 72 mm from the entry. A flush, loose ring fails the final retention inspection. For hardware, inspect both views and the retention of the ring. Use the GO and NO-GO checks in the physical protocol if the opening is ambiguous.

The controller triangulates the colored slug mark in two calibrated RGB views. Withdrawal is triggered by two consecutive observations of a drop greater than 23 mm with a ray mismatch below 6 mm. Tool contact, object poses, anchors and fracture notifications are unavailable to this decision. The recording contains independent ground truth for scoring and audit. Physics samples at 500 Hz, controls and cameras at 20 Hz.

## Run and replay

Build the pinned image with `python3 runtime/build_image.py` or use the configured YAM image. Mount this directory at `/task`, a writable output at `/outputs` and expose an RTX GPU.

```sh
/isaac-sim/python.sh /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none
/isaac-sim/python.sh /task/runtime/rescore.py --task /task/task.py /path/to/trajectory.jsonl --expect /path/to/result.json
```

The camera-driven YAM scored 1.0 in 240 actions (12.00 s) on Blackwell. The ring remained attached and the smaller entry had 20.76 mm of projected clearance. Blanking both camera inputs on the same setup gave 0.0. In a separate full episode with the same seed and task inputs, Dream-1 scored 0.0 after all 240 steps. The blade made no recorded contact with the enclosure or center disk, the disk stayed flush throughout and the outer ring remained intact. Independent replay reproduced both final scores.

For an independent model evaluation, run the same environment with the Dream policy, an authorized budget ledger and the appropriate credential. The earlier open-fixture Dream score belongs to the earlier environment.

