# Bimanual mouth grasp / lift verification

**No completed lift-and-three-second-hold success is claimed.** Folding and
clip actions are excluded. The most recent loaded recording is
`pinch-tilt-loaded-v74/mouth-grasp-lift-diagnostic.mp4` (6.9 seconds, 20 fps,
1280 x 720). It contains no interpolation or edited object positions.

The close-up makes robot geometry translucent and draws green/blue dots on
the actual original material vertices tracked by the left/right hand. Both
display aids are explicitly labelled. Numeric overlays distinguish material
rise, original point containment, total seating shift and hold creep.

| Run | Observation |
| --- | --- |
| v67 | 1,200 Hz rejected at frame 22 on a contents/film intersection. |
| v68 / v69 | Loaded / empty, strict protocol v1, stop during lift on slip/area criteria. They do not test a hold. |
| v70 / v71 | Deeper bite / inward motion, protocol v1. Also stop before a hold. v70's central material points are still in the pads, motivating separate seating/holding measurements. |
| v72 | Protocol v2: original point escape at frame 131. |
| v73 | Contents/film intersection at frame 63 before the tilt/lift executes. |
| v74 | Loaded, 35-degree tilt, stops at frame 137. Film rise 36.4-36.9 mm. Raw pad audit: left 5/5 points inside per wall; right 4/5 and 5/5. One point exceeds the main-pad footprint by about 29 micrometres. |
| v75 | Empty comparison, stops at frame 133; film rise 30.2-31.8 mm, right original point escape. |

The v74 footprint gate is conservative: a single point leaving the main-pad
rectangle is not equivalent to full physical grip loss, particularly with
other finger geometry present. The run stops there, so it does not establish
whether the remaining grasp would survive three seconds. No success should
be inferred from completed commands, the force readout, or this video alone.

All 16,440 v74 physics steps are observed with zero reported BOX/round/self
crossings and zero contact overflow. Cracker seam failures are zero in that
run. Observer limitations, uncalibrated materials and qualitative force
readout remain. Whole-bag suspension, food integrity grading, folding, sealing,
Arena integration and Dream inference are outside this component verdict.

`pinch-tilt-loaded-v74/states.npz` is the complete original 9.97 MiB recording.
Other runs retain selected endpoint states; their complete raw states remain
on Brev under `/data/robocurve/runs`. Each source archive was checked against
its isolated execution manifest. The v74 render manifest binds video, source,
trace and recording hashes; metadata and an encoded frame were inspected.

`raw_geometry_replay.py` reads original q/body states and actual main-pad
transforms, independently of robot cursor targets and logged rise summaries.
The v74 audit identifies original-reference frame 104, confirms hashes and
reports the terminal landmark counts. No hold frames exist; its verdict is
false. To rerun against the full v74 recording, extract its execution source
archive and supply its `experiments` directory as the second argument.

Focused tests: 12 pass under protocol v2, including empty/single-wall motion,
lost original landmarks, hold creep, missing frames and interrupted results.
Old v1 results and evaluators remain preserved without retrospective regrading.
