"""Reject opposing-part overlap throughout the authored clip joint range."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from gpu_contact_checks import rigid_boxes


class ClipGeometryTests(unittest.TestCase):
    def test_joint_range_has_no_opposing_box_overlap(self):
        root=ET.parse(Path(__file__).with_name("assets")/"clip_spring_v2.xml").getroot()
        lower=root.find("worldbody/body")
        upper=lower.find("body")
        owners,transforms,scales=[],[],[]
        def values(geom,name,default):
            return np.fromstring(geom.get(name,default),sep=" ")
        for owner,body in enumerate((lower,upper)):
            for geom in body.findall("geom"):
                owners.append(owner)
                transforms.append(np.r_[values(geom,"pos","0 0 0"),
                    Rotation.from_euler("xyz",values(geom,"euler","0 0 0")).as_quat()])
                scales.append(values(geom,"size","0 0 0"))
        pairs=np.array([(i,j) for i,a in enumerate(owners) for j,b in enumerate(owners) if a==0 and b==1])
        inputs=[wp.array(owners,dtype=int,device="cpu"),wp.array(np.array(transforms),dtype=wp.transform,device="cpu"),
                wp.array(np.array(scales),dtype=wp.vec3,device="cpu"),wp.array(pairs,dtype=wp.vec2i,device="cpu")]
        count=wp.zeros(1,dtype=int,device="cpu");flags=wp.zeros(len(pairs),dtype=int,device="cpu")
        low,high=np.fromstring(upper.find("joint").get("range"),sep=" ")
        for angle in [*np.linspace(low,high,81),.1]:
            with self.subTest(angle=angle):
                body=wp.array([[0,0,0,0,0,0,1],np.r_[[0,0,0],Rotation.from_euler("y",angle).as_quat()]],
                              dtype=wp.transform,device="cpu")
                count.zero_()
                wp.launch(rigid_boxes,len(pairs),inputs=[body,*inputs,0,count,flags],device="cpu")
                hits=int(count.numpy()[0])
                if angle==.1:
                    self.assertGreater(hits,0,"overclosing negative control should intersect")
                else:
                    self.assertEqual(hits,0)


if __name__=="__main__":
    unittest.main()
