"""Attribute a sector's over-centre motion to the strike that immediately preceded it."""
class StrikeCredit:
    def __init__(self):
        self.last_strike = [-100] * 8
        self.last_depth = [0.] * 8
        self.attributed = set()

    def update(self, step, depth_mm, fast_impulses, grasp):
        for i, depth in enumerate(depth_mm):
            if depth <= 2.:
                self.attributed.discard(i)
            if grasp and fast_impulses[i] > .012:
                self.last_strike[i] = step
            if self.last_depth[i] <= 2. < depth and step-self.last_strike[i] <= 1 and grasp:
                self.attributed.add(i)
            self.last_depth[i] = depth
        return [i in self.attributed and d >= 3. for i,d in enumerate(depth_mm)]
