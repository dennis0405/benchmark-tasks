"""Score shared-runner observations from recorded contact and geometry evidence."""
import math


class BridgeScorer:
    def __init__(self):
        self.last_step = 0
        self.held_frames = 0
        self.holding_tool = None
        self.violation = False

    def update(self, *, step, objects, root_poses, robot, mechanics=None):
        if type(step) is not int or step <= self.last_step:
            raise ValueError("steps must increase strictly")
        gap = step != self.last_step + 1
        self.last_step = step
        if gap:
            self.held_frames = 0
            self.holding_tool = None
        if not isinstance(mechanics, dict) or type(mechanics.get("violation")) is not bool or type(mechanics.get("clear")) is not bool:
            self.held_frames = 0
            self.holding_tool = None
            return {"score": 0., "success": False, "valid": False}
        self.violation |= mechanics["violation"]
        tools = mechanics.get("tools")
        if not isinstance(tools, dict) or set(tools) != {"clip", "peg"}:
            self.held_frames = 0
            self.holding_tool = None
            return {"score": 0., "success": False, "valid": False}
        partial, holding = False, None
        for name in ("clip", "peg"):
            entry = tools[name]
            if not isinstance(entry, dict):
                self.held_frames = 0
                self.holding_tool = None
                return {"score": 0., "success": False, "valid": False}
            flags = [entry.get(k) for k in ("folded", "aligned", "jaw_on_layers", "still", "contact")]
            overlap = entry.get("overlap")
            if any(type(v) is not bool for v in flags) or type(overlap) not in (float, int) or not math.isfinite(overlap) or overlap < 0:
                self.held_frames = 0
                self.holding_tool = None
                return {"score": 0., "success": False, "valid": False}
            qualifies = flags[0] and flags[1] and flags[2] and flags[4]
            partial |= qualifies
            if qualifies and flags[3] and mechanics["clear"] and overlap >= .096:
                holding = holding or name
        if self.violation:
            self.held_frames, self.holding_tool = 0, None
            return {"score": 0., "success": False, "valid": True, "dropped": True, "held_frames": 0}
        if holding is None:
            self.held_frames, self.holding_tool = 0, None
        elif holding == self.holding_tool:
            self.held_frames += 1
        else:
            self.held_frames, self.holding_tool = 1, holding
        score = 1. if self.held_frames >= 60 else .5 if partial else 0.
        return {"score": score, "success": score == 1., "valid": True,
                "clear": mechanics["clear"], "dropped": False,
                "held_frames": self.held_frames, "holding_tool": self.holding_tool}
