# Set and lock an angular gripper's opening stop

Status: design input for the selected production task. This is not a validated
environment, a full-score demo or the frozen TaskContract. No native success or
Dream result is claimed. The production contract and measured native results
must supersede this draft before release.

## Industrial operation

An angular gripper needs its opening stop set before robot commissioning or a
change of tooling. The SCHUNK DRG manual describes fitting an adjusting screw,
setting the required opening, securing it with nut 73, then testing with air.
Adjustment takes place with air off and the gripper closed and unpressurised;
corrections also require air off before adjusting again.

Primary reference: [SCHUNK DRG manual 389041, version 05.00, section 5.4.1,
printed page 28](https://media.distributordatasolutions.com/web_schunk/2023q1/documents/ea9d09a544db3115509a083c4761f6a0e4de75cd.pdf).
This task represents that adjustment/locking operation with an explicitly
specified bench fixture; it does not claim that the first-party CAD is the
manufacturer's exact product or that its proposed coefficients are calibrated.

## Intended instruction

Set the angular gripper's maximum opening to 90 degrees and secure the stop.
Turn off and vent the air before adjusting. Use the driver to control the stop
screw while tightening its jam nut with the wrench. Remove the tools and test
the opening through repeated open/close cycles. Leave the adjustment retained,
the tools parked and the bench safe.

Equivalent safe physical strategies receive the same outcome score. A hand
identity, a specific waypoint sequence or mandatory counterholding narrative
must not substitute for actual physical retention.

## Starting setup

- A full-scale representative angular-gripper fixture bolted to the table,
  with two hinged jaws, a piston/stop transmission and the air supply vented.
- The adjustment screw and nuts are already fitted. Accessory installation is
  outside this task. The lower locking nut is loose; the initial opening stop
  is outside the target range.
- A separate hex driver and thin open-ended wrench, resting in reproducible
  table locations. Both remain free dynamic tools in production.
- A reproducible valve or test-fixture control for the function test. Its
  physical interface and actuation model must be specified in the final task.

The proposed geometry is in `fixture/fixture-proposal.json` and `geometry.json`.
All dimensions, masses, internal transmission, thread lead and friction/force
parameters in these seeds are benchmark assumptions. The source manual's
20–180 degree range concerns the adjustable maximum opening; ordinary closing
below 20 degrees remains allowed.

## Outcome and scoring intent

The target is total opening of 90 degrees, provisionally within 3 degrees,
retained over three released open/close cycles with at most 2 degrees drift.
These numerical choices are benchmark design inputs, not manufacturer values.
The final contract must freeze the exact measurement, dwell and revocation rules.

Partial credit should correspond to a genuine adjusted state or a completed
part of the functional verification. A hand-supported angle, a tool wedged in
the mechanism or a historical target reached before a later damaging event
must not masquerade as retained calibration. Full credit requires the locking
nut bearing on its seat, repeated function, and measured hand/tool clearance.
Define physically available unsafe adjustment, loss and damage outcomes
explicitly; do not invent force/strength measurements absent from the model.

## Required causal model and validation

Locknut turning must transfer reaction torque into the adjusted screw. A real
seated tool can oppose it; an unseated tool cannot. The retained outcome must
come from the declared threads/contact/friction, with no goal-angle servo,
post-initialisation pose writes or scripted locked-state shortcut.

Validate components and reset first, then build the ordinary 14-D YAM demo.
Record restrained/unrestrained and wrong-seating controls, lock/release and
retention, actual simulation/control rates and timestep refinement. Final
source must pass two fresh-process YAM demos, relevant scorer controls,
independent rescore and an isolated-copy rerun. Only then run the one allowed
Dream episode and repeat the full novelty check on the actual built task.
