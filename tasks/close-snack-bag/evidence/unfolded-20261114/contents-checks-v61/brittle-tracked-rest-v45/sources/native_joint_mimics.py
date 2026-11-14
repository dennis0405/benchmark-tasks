"""Translate imported legacy linear mimic rows to Newton 1.7 joint metadata.

The unchanged authored relationship q0=a+b*q1 is oriented so the passive
finger follows the actuated finger. No state is prescribed by this adapter.
"""
import numpy as np


def configure(model):
    if not model.constraint_mimic_count:return []
    leader=model.joint_mimic_joint.numpy().copy()
    coefficients=model.joint_mimic_coeffs.numpy().copy()
    ke=model.joint_target_ke.numpy();start=model.joint_qd_start.numpy()
    rows=zip(model.constraint_mimic_joint0.numpy(),model.constraint_mimic_joint1.numpy(),
             model.constraint_mimic_coef0.numpy(),model.constraint_mimic_coef1.numpy(),
             model.constraint_mimic_enabled.numpy())
    converted=[]
    for j0,j1,a,b,enabled in rows:
        if not enabled:continue
        j0,j1=int(j0),int(j1)
        active0=ke[int(start[j0])]>0;active1=ke[int(start[j1])]>0
        if active0==active1:raise ValueError("Mimic conversion requires one driven and one passive joint")
        if active0:
            if abs(b)<1e-12:raise ValueError("Cannot invert constant mimic relationship")
            follower,reference,offset,multiplier=j1,j0,-float(a)/float(b),1./float(b)
        else:follower,reference,offset,multiplier=j0,j1,float(a),float(b)
        if leader[follower] not in (-1,reference):raise ValueError("Conflicting authored mimic metadata")
        leader[follower]=reference;coefficients[follower]=[offset,multiplier]
        converted.append(dict(follower=follower,reference=reference,offset=offset,multiplier=multiplier))
    model.joint_mimic_joint.assign(leader)
    model.joint_mimic_coeffs.assign(coefficients)
    return converted
