# Hydraulic model and evidence limits

The circuit has about 8.042 L of water in 10 m of 32 mm ID copper tube.
At20 C, water compressibility46.5ppm/atmosphere gives 0.369mL/bar.
A closed-ended thin-wall estimate using E130GPa,nu0.343,wall1.5mm gives
0.120mL/bar additional pipe compliance. Their sum0.489mL/bar
supports a rounded candidate value0.50mL/bar. This is an estimate, not measured
calibration; coiling, soldered joints, actual alloy and temperature are simplified.
The pump/manifold effective0.10mL/bar is nominal and remains unmeasured.

Sources:
- https://nvlpubs.nist.gov/nistpubs/jres/097/jresv97n3p335_a1b.pdf
- https://emtoolbox.nist.gov/publications/nationalstandardslaboratorytechnicalpaperno25.pdf

Volume change is V*p*D/(E*t)*(1.25-nu) for closed ends with small strain.
Pressure is gauge pressure. Pressure-volume stored energy is C*p^2/2;
bar*mL is0.1J. Cylinder compression work acts on the native lever as p*dV/dtheta.
The outlet check passes only forward flow. A return stroke releases cylinder
elastic storage then draws water through the inlet; it cannot suck the pipe empty.
The isolation valve disconnects the test circuit. Opening the return valve alone
cannot vent the isolated circuit. Both valves must reopen to vent the test side.

All volume-transfer laws use physical substep dt, not accelerated time.
A0.006mL/(s*bar) nominal leak gives exponential test-side pressure decay with
time constant0.50/0.006=83.33s when isolated. It is a test-specimen parameter;
no measured real fitting leak curve is claimed. A real fixture needs its leak
rate measured before reproducing those exact pressure trajectories.

No source uses the scripted action, phase, goal or score to generate pressure.
Physics writes only reaction torque; no object transforms or robot joints are
written to simulate manipulation. The noncolliding pressure needle is a display.
