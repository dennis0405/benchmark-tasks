"""Lumped, isothermal hydrostatic circuit; SI internally except labelled bar/ml.

This is a candidate model, not calibrated apparatus data. Measured pump/valve
motion supplies inputs. No action, target pose, demo clock or score is an input.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    stroke_ml: float = 16.0
    stroke_angle_rad: float = math.radians(50)
    circuit_compliance_ml_bar: float = 0.50
    pump_compliance_ml_bar: float = 0.10
    leak_ml_s_bar: float = 0.006
    relief_bar: float = 5.0
    reservoir_ml: float = 4500.0


class Hydraulics:
    def __init__(self, params=None):
        self.p = params or Parameters()
        self.reset()

    def reset(self, angle=math.radians(30)):
        self.angle = angle
        self.pump_bar = self.test_bar = 0.0
        self.pumped_ml = self.leaked_ml = self.vented_ml = 0.0
        self.net_displaced_ml = self.intake_ml = 0.0
        self.delivered_ml = 0.0
        self.relief_ml = 0.0
        self.peak_bar = 0.0

    def advance(self, dt, angle, isolation_open, drain_open):
        if not 0 < dt <= .1:
            raise ValueError('Advance at physical substeps, not accelerated test time')
        if not all(math.isfinite(v) for v in (angle, isolation_open, drain_open)):
            raise ValueError('Nonfinite measured mechanism')
        p = self.p
        angle=max(0.,min(p.stroke_angle_rad,angle))
        # Signed cylinder displacement. An upstroke first releases compressed
        # cylinder water, then draws fresh water at zero gauge pressure. The
        # outlet check prevents the TEST circuit from powering the return.
        delta = math.sin(self.angle)-math.sin(angle)
        dv = p.stroke_ml*delta/math.sin(p.stroke_angle_rad)
        self.angle = angle
        self.pumped_ml += max(0.,dv)
        self.pump_bar += dv/p.pump_compliance_ml_bar
        makeup=max(0.,-self.pump_bar*p.pump_compliance_ml_bar)
        if self.intake_ml+makeup > p.reservoir_ml:
            raise RuntimeError('Reservoir empty')
        self.intake_ml+=makeup
        self.pump_bar=max(0.,self.pump_bar)
        self.net_displaced_ml+=dv+makeup
        cp, ct = p.pump_compliance_ml_bar, p.circuit_compliance_ml_bar
        # Exact two-volume equalization, stable even when the isolation valve
        # opens widely. Keep pressure on the TEST side of the isolation valve.
        conductance = 25.*max(0., min(1., isolation_open))**2
        if conductance > 0 and self.pump_bar > self.test_bar:
            test_before = self.test_bar
            mean = (cp*self.pump_bar+ct*self.test_bar)/(cp+ct)
            diff = (self.pump_bar-self.test_bar)*math.exp(-conductance*(1/cp+1/ct)*dt)
            self.pump_bar = max(0., mean+ct/(cp+ct)*diff)
            self.test_bar = max(0., mean-cp/(cp+ct)*diff)
            self.delivered_ml += ct*(self.test_bar-test_before)
        leaked = self.test_bar*ct*(1-math.exp(-p.leak_ml_s_bar/ct*dt))
        self.test_bar -= leaked/ct
        self.leaked_ml += leaked
        # Return valve is downstream of the outlet check. Reopening isolation
        # connects the test circuit to that return; a closed isolation leaves
        # test pressure trapped even with return open.
        vent = self.pump_bar*cp*(1-math.exp(-15.*max(0., min(1., drain_open))**2/cp*dt))
        self.pump_bar -= vent/cp
        self.vented_ml += vent
        test_vent=self.test_bar*ct*(1-math.exp(-15.*min(max(0.,isolation_open),max(0.,drain_open))**2/ct*dt))
        self.test_bar-=test_vent/ct
        self.vented_ml+=test_vent
        excess = max(0., self.pump_bar-p.relief_bar)*cp
        self.pump_bar -= excess/cp
        self.relief_ml += excess
        self.peak_bar = max(self.peak_bar, self.test_bar, self.pump_bar)
        return self.state()

    def state(self):
        return dict(test_bar=self.test_bar, pump_bar=self.pump_bar,
                    pumped_ml=self.pumped_ml, delivered_ml=self.delivered_ml, leaked_ml=self.leaked_ml,
                    net_displaced_ml=self.net_displaced_ml,intake_ml=self.intake_ml,
                    vented_ml=self.vented_ml, relief_ml=self.relief_ml,
                    peak_bar=self.peak_bar)

    def resisting_torque(self, angle):
        # p*dV/dtheta; bar*ml = 0.1 joule. Cylinder pressure acts both ways;
        # on return its finite stored energy is released before the inlet opens.
        return self.pump_bar*.1*self.p.stroke_ml*math.cos(angle)/math.sin(self.p.stroke_angle_rad)
