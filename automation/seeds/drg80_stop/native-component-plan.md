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
