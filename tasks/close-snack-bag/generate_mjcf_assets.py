"""Write isolated MuJoCo sources for Isaac Lab's MJCF-to-USD converter."""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

here = Path(__file__).resolve().parent
out = here / "asset_sources"
out.mkdir(exist_ok=True)
template = ET.parse(here / "prototype.xml").getroot()


def write_asset(name, body):
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "default")
    world = ET.SubElement(root, "worldbody")
    body.set("pos", "0 0 0")
    world.append(body)
    ET.ElementTree(root).write(out / f"{name}.xml")


world = template.find("worldbody")
write_asset("bag_v4", copy.deepcopy(world.find("body[@name='bag']")))
clip = copy.deepcopy(world.find("body[@name='clip']"))
write_asset("clip_v5", copy.deepcopy(clip))
clip.set("name", "peg")
for geom in clip.iter("geom"):
    geom.set("rgba", "0.65 0.36 0.16 1" if "lower" in geom.get("name") else "0.82 0.58 0.30 1")
write_asset("wide_peg_v5", clip)

ring = ET.Element("body", name="band")
ET.SubElement(ring, "freejoint")
for i in range(16):
    a, b = 2*math.pi*i/16, 2*math.pi*(i+1)/16
    coords = (0.026*math.cos(a), 0.019*math.sin(a), 0, 0.026*math.cos(b), 0.019*math.sin(b), 0)
    ET.SubElement(ring, "geom", name=f"segment{i}", type="capsule", size="0.0012",
                  fromto=" ".join(map(str, coords)), mass="0.0001", rgba="0.85 0.16 0.12 1")
write_asset("band_v3", ring)

paper = ET.Element("body", name="paperclip")
ET.SubElement(paper, "freejoint")
for i, coords in enumerate(((-.012, -.005, 0, .012, -.005, 0), (.012, -.005, 0, .012, .005, 0),
                            (.012, .005, 0, -.009, .005, 0), (-.009, .005, 0, -.009, .001, 0),
                            (-.009, .001, 0, .007, .001, 0))):
    ET.SubElement(paper, "geom", name=f"wire{i}", type="capsule", size="0.0008",
                  fromto=" ".join(map(str, coords)), mass="0.0002", rgba="0.65 0.66 0.69 1")
write_asset("paperclip_v3", paper)
