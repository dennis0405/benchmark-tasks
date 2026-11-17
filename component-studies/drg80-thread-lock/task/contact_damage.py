"""Read-only manifold damage accounting at the benchmark's native substep rate."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ContactPoint:
    body_a: int
    body_b: int
    signed_separation_m: float
    compressive_impulse_ns: float
    relative_normal_speed_m_s: float


class ContactDamage:
    def __init__(self, hz, body_ids, intended, exempt):
        if hz not in (240, 480):
            raise ValueError("unsupported physics rate")
        self.hz = hz
        self.ids = frozenset(body_ids)
        self.intended = {tuple(sorted(pair)) for pair in intended}
        self.exempt = {tuple(sorted(pair)) for pair in exempt}
        if self.intended & self.exempt:
            raise ValueError("intended and exempt pairs must be disjoint")
        self.active = {}
        self.tick = 0
        self.failure = None

    def observe(self, tick, points):
        if tick != self.tick + 1:
            raise ValueError("missing or reordered contact stream")
        self.tick = tick
        pairs = {}
        for p in points:
            pair = tuple(sorted((p.body_a, p.body_b)))
            if pair[0]==pair[1] or not self.ids.issuperset(pair):
                raise ValueError("unregistered contact pair")
            if not all(map(isfinite, (p.signed_separation_m,p.compressive_impulse_ns,p.relative_normal_speed_m_s))):
                self.failure = self.failure or (tick,pair,"non-finite")
                continue
            if pair in self.exempt:
                continue
            impulse,penetration,work = pairs.get(pair,(0.,0.,0.))
            j = max(0.,p.compressive_impulse_ns)
            pairs[pair] = (impulse+j,max(penetration,-p.signed_separation_m),
                           work+j*max(0.,-p.relative_normal_speed_m_s))
        for pair in set(pairs)|set(self.active):
            j,pen,added = pairs.get(pair,(0.,0.,0.))
            event,zeros = self.active.get(pair,(0.,0))
            if j>0:
                event+=added
                self.active[pair] = event,0
            elif pair in self.active:
                if zeros>=1:
                    del self.active[pair]
                    event=0.
                else:
                    self.active[pair]=event,zeros+1
            max_force,max_work = (250.,8.) if pair in self.intended else (150.,5.)
            for name,value,limit in (("penetration_m",pen,.002),("force_n",j*self.hz,max_force),("event_work_j",event,max_work)):
                if value>limit and self.failure is None:
                    self.failure = (tick,pair,name)
        return self.failure
