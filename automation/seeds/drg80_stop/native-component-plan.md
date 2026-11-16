# Screw/nut component plan, not a physics result

Status: optional first-party design input. No native solver run, grasp, torque
transfer, score or YAM success is established here. The numerical contract and
native measurements take precedence. Do not count this document as evidence of
an environment pass.

## Causal behavior to establish

Turning the nut must apply a reaction to the adjusted screw through the thread
interface. A held driver can oppose that reaction through real tool contact.
When the nut seats on the cover, preload should create retention. This is the
mechanical distinction evaluated against the belt-mount task: holding against
an existing belt load is not a model of concentric screw/nut coupling.

## Native joint/contact route worth testing

An economical coupon can omit the complete gripper and use these rigid parts:

1. A fixed cover with a genuine through bore and a nut-bearing surface.
2. A screw rotor on a native revolute joint.
3. An intermediate axial carrier on a prismatic joint relative to the screw.
   It represents a joint coordinate, not a visible extra hand or a fake grasp.
4. A nut on a revolute joint relative to that carrier. Relative angular drive
   damping can produce an implicit, equal-and-opposite coupling to the screw.
5. The actual nut face contacting the cover, rather than a software "locked"
   flag holding a pose. Both tools remain independent free rigid bodies.

The relative angle and thread lead can define an axial spring-drive target for
the carrier. After seating, spring compression loads the nut against the cover.
Native contact friction is a candidate for static retention. Crucially, allow
axial travel beyond the contact plane: if the guide's hard limit carries the
normal load first, the bearing surface can have essentially no friction load.

A conservative helical spring model has an extension such as
`e = (z_n - z_s) + lead * (theta_n - theta_s) - e0` and potential
`U = 0.5 * k * e**2`. Derive consistent axial forces and opposing torques from
this potential; do not attach an arbitrary success torque to the tool. If using
native implicit drives for the axial coordinate, verify the discrete coupling
and actual angular-drive unit conversions in the pinned backend. Coefficients
are not supplied as measured industrial values. Explicit high-gain friction
evaluated only at 20 Hz can be unstable on these small inertias.

Screw axial motion, a self-locking thread approximation, and the gripper's stop
transmission need separate declared treatment. An analytic stop limit must
depend on the measured mechanism coordinate rather than the desired score.
If an idealization omits load feedback, disclose it and test the relevant
loaded behavior. No pose writes after initialization and no hidden tool welds.

## Probes before the YAM controller

- Check actual pinned-USD loading, guide freedom, native contact, gravity and
  friction; the CPU-only asset checks do not establish any of these.
- Apply controlled diagnostic nut torque with a free screw, then with a real
  restraint. Record native screw/nut angles, gap, forces and release behavior.
- Seat and release the nut, measure retention, then loosen it and measure the
  resulting behavior. An unlocked thread need not backdrive under axial load;
  do not force an angle failure solely to make a negative control look decisive.
  Missing nut seating can fail the required secured outcome even if the screw
  temporarily retains its setting through thread friction. Check timestep
  sensitivity at the declared sample rate.
- Test the real hex and wrench interfaces; off-axis or unseated tools must not
  transmit an invented rotation merely because a hand is nearby.
- Once adopted, connect the same component model to both scripted and Dream
  execution. The read-only scorer observes native state; it never actuates it.

An uncompensated, unrestrained tightening control may miss the final setting.
This does **not** justify forbidding physically successful alternatives such as
safe precompensation. Grade the actual retained outcome and declared industrial
requirements, not mandatory hand identity or a preferred counterholding plan.
If a simpler valid strategy works, document it and reassess the actual tested
ability in final novelty review.

## Axial self-locking is distinct from fastening reaction

[igus](https://toolbox.igus.com/motion-plastics-blog/self-locking-what-is-it/)
explains that self-locking depends on lead/thread geometry, friction and
operating conditions. [Thomson](https://www.thomsonlinear.com/en/support/tips/is-it-possible-to-backdrive-an-actuator)
also distinguishes self-locking screw designs from backdrivable systems.
These are general engineering references, not measurements of the SCHUNK unit
or a license to copy another product's friction coefficients.

Therefore measure fastening-induced screw rotation separately from response
to axial piston load. Validate the assumed friction regime and report what the
unrestrained control actually does. If real countertorque/setpoint coupling is
absent, revise the ability claim and final novelty assessment; do not invent
an unstable screw merely to require a particular two-handed strategy.


## Alternative implicit helical coupon to test first

The pinned `generatedSchema.usda` contains `class "PhysxMimicJointAPI"`
(the class name is quoted). Its presence is not a successful runtime test.
NVIDIA's [107.3 articulation guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joints)
describes prismatic/revolute coupling within one articulation, with the equation
`q_A + G*q_B + offset = 0` and optional implicit compliance. This offers a
possible route that avoids explicit high-gain force updates at the policy rate.

For an isolated coupon, a fixed frame can connect through a shape-free axial
carrier (prismatic) and then a screw rotor (revolute). Couple the rotor to the
carrier using a rotational mimic follower. For the nut, a second carrier and
rotor relative to the screw give relative axial/rotational thread coordinates.
Both pairs must belong to the SAME articulation. Their coordinates are
mechanical freedoms, not tool attachments. Keep the driver/wrench free dynamic
bodies and the cover/nut-bearing collision active. Nut-to-cover contact needs
self-collision configured for the articulation; do not disable that essential
contact to make parsing or the demo succeed.

Verify native one-turn axial travel and both applied-torque/load reactions
before selecting the gear sign or interpreting USD degree/radian conversions.
Check angular wrap behavior and enough real guide travel for the required
turns; a wrapped coordinate must not generate a false thread jump. Do not
assume a smooth bore alone implements a helical thread. Load-dependent thread
friction is separate from the ideal helix and bearing contact: without a
justified friction approximation, an ideal screw can backdrive unrealistically.
Any coefficients remain benchmark assumptions pending measurement.

This is an implementation lead, not an adopted mechanism or native pass. Test
the smallest coupon first; use the simplest route that actually preserves the
measured reaction, seating and retained outcome. Avoid adding an unsupported
new dependency or changing the official policy/robot action interface.

## Observed direct-GPU incompatibility, 2026-11-16

The actual `drg80-component-preflight-38c712934d.log` in production run
`20261114T031727-dc73f2bb` reports that the prototype's three gear/rack coupling
constraints cannot be added to a scene configured for direct GPU access, and
requests a D6 joint instead. That invocation produced no valid component report.
Therefore the existing composed gear/rack proposal is **not a validated GPU
mechanism**. Do not reuse its authoring/schema validity as runtime evidence.

An isolated diagnostic may deactivate these unsupported constraints to identify
which ordinary joints/bodies load correctly, but cannot thereby establish
lever-to-piston/jaw coupling or qualify the task. A replacement must restore the
physical transmission and load reaction before the frozen component checks,
YAM demo or Dream evaluation. D6/articulation-mimic alternatives described above
remain leads to test, not confirmed replacements. The GPU contact/callback cube
probe passed independently; it does not resolve this mechanism incompatibility.

`mimic_api_probe.py` now preserves the exact source of a successful isolated
GPU articulation test, SHA256
`384ff268d6f1071a81210e069ec95b582ef1c8b6007da792e7115e71641028a0`.
The trusted controller recorded13 passing checks in trial
`38efb0ce-5ac1-43ff-80ff-1bff4ffb3ca8`, under
`artifacts/environment-preflights.json` with `qualification:false`. Its banner
and scope remain deliberately limited to discovery, never full task validation.

The actual forward motion was6.97562482rad and1.11013651mm; reverse motion was
-6.59552331rad and-1.04969740mm. Measured lead error relative to1mm/revolution
was6.18464e-8m/revolution; maximum all-sample helix residual was0.000945033rad.
There were three native links and two DOFs in one articulation at240Hz.
The authored gearing was-360000 degrees/metre, with observed body rotations
converted to radians in analysis. These are isolated native observations, not
measurements or calibration of the actual SCHUNK equipment.

The first motor-command attempt remained stationary because updating USD drive
targets after reset invokes a CPU-side setter that direct-GPU mode prohibits.
The passing source instead reads the native articulation DOF names and calls
`set_dof_velocity_targets` with an explicit Warp uint32 articulation index.
Its explicit output paths and `--visualizer none` avoid the earlier CLI/path
failures. The real source and report hashes were compared before preserving it.

This ideal, gravity-free, noncontact coupon does not establish thread friction,
the frozen compliant constitutive law, preload, full gripper transmission or
robot success. Reusing it in a changed runtime still requires source review
and actual execution; import/compilation is not a physics test.

## Loaded experiments: force transfer is not full accuracy

The first loaded hard-mimic test failed: projected DOF forces were near zero
and the reported angular/linear residual reached1.234rad. That force getter
could not certify the intended transmitted load in this setup. Do not turn
the measured zero into proof of no physical load, or replace it with a
rotation ratio or spring calculation labelled as measured native force.

A later known-pair physical contact test isolated the carrier collider from
the overlapping frame/rotor diagnostic geometry. Source SHA256
`57cee70840fce1409f45869f8ab9dca10a0c08176644bb4e63c65a975f0f9b93`
produced20 passing checks out of21, with actual filtered contact medians
31.419628N and31.413954N under signed0.005Nm inputs. The ideal prediction was
31.415926N. Force/impulse scaling and filtered-versus-net force also matched.
However, the evaluated loaded-window angular residual was0.0595967rad,
exceeding0.02rad, and the full trace contained a0.107198rad transient.
This is a **failed accuracy profile**, with useful scoped force observations.

A separate seated-tool composite-inertia profile still failed the same
accuracy test at0.0706611rad. Both original reports/traces remain preserved.
Never transfer a composite tool inertia into the bare screw, duplicate tool
mass, or weld a real tool to create apparent stability. The full task must
include actual tool contact and tool-released checks.

The next proposed route is the already frozen finite constitutive law, not
continued inertia tuning of an ideal coupling. The
[official fixed-tendon example](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#fixed-tendons)
exposes stiffness, damping and weighted joint coordinates. It is a native
implementation lead only; mixed units and conservative force coefficients
still require an installed-SDK source check and actual loaded experiment.
For a finite spring, elastic displacement is expected: compare native contact
reaction with the unchanged frozen force law. Do not relabel the old ideal
accuracy failures or claim full qualification from a different coupon.
