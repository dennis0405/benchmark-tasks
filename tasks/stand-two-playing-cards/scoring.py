"""Count only continuous self-support. Contact and geometry are measured, with no phase labels."""
class Scorer:
    def __init__(self):
        self.last = 0
        self.hold = 0
        self.seated_frames = 0

    def update(self, *, step, objects, root_poses, robot, mechanics):
        if step != self.last+1:
            raise ValueError('Missing sample in release interval')
        self.last = step
        seated = (mechanics['ridge_contact'] and all(mechanics['on_mat'])
                  and mechanics['ridge_error_m'] < .006
                  and min(mechanics['inward_lean']) > .07
                  and max(mechanics['inward_lean']) < .87
                  and mechanics['edge_overlap_m'] > .02
                  and max(mechanics['base_error_m']) < .0025
                  and mechanics['holders_clear'])
        stationary = (mechanics['vmax'] < .012 and mechanics['wmax'] < .16)
        self.seated_frames = self.seated_frames + 1 if seated else 0
        independent = seated and stationary and mechanics['hands_clear']
        self.hold = self.hold+1 if independent else 0
        # At 20 Hz, 11 observations span 0.5 s and 61 observations span 3 s.
        ok = self.hold >= 61
        partial = self.seated_frames >= 11
        return dict(score=1. if ok else .5 if partial else 0.,
                    success=ok, held_frames=self.hold, seated_frames=self.seated_frames, clear=mechanics['hands_clear'])
