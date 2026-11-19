import math
import numpy as np

BASE = np.array([.32, 0., .765])
WIDTH, HEIGHT, SKEW = .40, .30, 7.
a = math.radians(SKEW)
corners = np.array([[0.,0.],[WIDTH,0.],[WIDTH+HEIGHT*math.sin(a),HEIGHT*math.cos(a)],[HEIGHT*math.sin(a),HEIGHT*math.cos(a)]])
corners -= corners.mean(axis=0)
yaw = math.pi/2-math.atan2(*(corners[2]-corners[0])[::-1])
c, s = math.cos(yaw), math.sin(yaw)
corners = corners @ np.array([[c,s],[-s,c]])
HALF = np.linalg.norm(corners[2]-corners[0])/2 + .014+.006+.008
START_X = -.105
GRIP_Z = .102
