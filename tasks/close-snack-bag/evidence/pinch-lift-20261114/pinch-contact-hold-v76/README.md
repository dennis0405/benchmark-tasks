# Continue past pad-edge flags: actual release observed

The user explicitly requested continuation unless the bag is physically
released or intersecting. A fresh v3 episode keeps the same material, contents,
gripper effort, 50 mm lift, 35-degree wrist tilt and all-substep geometry
checks. Footprint escape, tracked-point displacement and target tracking
errors are diagnostic, rather than reasons to stop the hold.

The run stops at frame 148 on actual right-hand release. At steps 144-148,
nearest film-particle/finger distance is 19.72, 28.54, 28.26, 28.26 and 28.31 mm;
both opposed contact readouts are zero throughout. At the terminal state the
left hand is also clear (24.23 mm), with zero contact. The raw q/body-geometry
audit independently confirms the right-hand separated interval and matches
logged material rise/distances. This is not a microscopic footprint failure.

Three commanded hold frames occur after release begins; they are not physical
retention. No three-second bimanual raised hold is observed. The verdict is
**failed**, not unassessed due to the former 0.03 mm boundary rule.

- Every one of 17,760 physical steps has zero reported film/BOX, round-shape or
  transverse-self intersections and no contact-capacity overflow.
- One predefined cracker seam breaks. Material and force calibration, food
  integrity grading, full-bag suspension, folding and sealing remain separate.
- `contact-hold-diagnostic.mp4` is 7.45 s, 149 frames at 20 fps, 1280 x 720.
  The close-up transparently displays the robot and original material dots.
  The video is 1x recorded physics, without interpolation or a frozen hold.
- `states.npz` is the complete original recording (10.76 MiB). Full traces,
  colliders, geometry, runtime results, command arguments and hashes are here.
- `execution-source.tar.gz` was verified against its isolated manifest.
  `render_unfolded.py` is the separately hashed renderer. `verify_contact_hold.py`
  and `raw-contact-hold-assessment.json` preserve the independent audit.
- Fifteen focused tests pass, including retained contact despite boundary
  flags, genuine physical separation and contact near the end of a long pad.

The contact distance is measured to physical BOX/CAPSULE/SPHERE finger
colliders using cloth particles, not a continuum surface or a calibrated
contact sensor. Existing simulation/geometry exclusions remain. No model
inference, public task score or submission is claimed.
