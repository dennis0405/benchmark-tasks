"""Object servo regression: retain commanded bias under a loaded hand."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from clip_controller import move_clip_root


class LoadedHand:
    """A free grasp following a hand with a constant 0.8 mm tracking offset."""
    def __init__(self,reachable=True):
        self.offset=np.array([-.06,0.,0.])
        self.tcp=np.array([.54,0.,1.04]);self.rotation=Rotation.identity()
        self.panel=SimpleNamespace(cursor_pos=self.tcp.copy(),cursor_quat=self.rotation.as_quat())
        self.panel.ik=SimpleNamespace(ee_pose=lambda state:(self.tcp.copy(),self.rotation.as_quat()))
        self.session=SimpleNamespace(arms={"right":self.panel})
        self.sim=SimpleNamespace(state=None)
        self.film=None
        self.takes=SimpleNamespace(_fold_jaws=lambda film,arm:self.tcp.copy())
        self.failure=None;self.marks={};self.phase="loaded_test";self.reachable=reachable

    def pose(self):
        return np.r_[self.tcp-self.rotation.apply(self.offset),self.rotation.as_quat()]

    def move(self,points,quats,*args,**kwargs):
        self.panel.cursor_pos=np.asarray(points["right"]).copy()
        self.panel.cursor_quat=np.asarray(quats["right"]).copy()
        self.tcp=self.panel.cursor_pos-[0,0,.0008]
        self.rotation=Rotation.from_quat(self.panel.cursor_quat)
        yield
        return self.reachable


class ClipServoTests(unittest.TestCase):
    def test_tracking_bias_is_integrated_instead_of_reintroduced(self):
        c=LoadedHand();target=np.array([.576,0.,.9765])
        grasp=dict(offset=c.offset.copy(),relative_rotation=Rotation.identity())
        with patch("clip_controller.clip_pose",lambda controller:controller.pose()):
            list(move_clip_root(c,target,Rotation.identity(),grasp,precise=True))
        self.assertIsNone(c.failure)
        self.assertLess(np.linalg.norm(c.pose()[:3]-target),.0006)

    def test_failed_hand_motion_stops_object_route(self):
        c=LoadedHand(reachable=False)
        grasp=dict(offset=c.offset.copy(),relative_rotation=Rotation.identity())
        with patch("clip_controller.clip_pose",lambda controller:controller.pose()):
            list(move_clip_root(c,np.array([.576,0.,.9765]),Rotation.identity(),grasp,precise=True))
        self.assertIn("right hand did not reach",c.failure)


if __name__=="__main__":unittest.main()
