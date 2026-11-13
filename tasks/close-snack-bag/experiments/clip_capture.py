"""Independent measured film capture between the actual spring jaws.

No force inference: this reports geometry, with an explicit 0.3 mm contact
skin tolerance. It must be combined with observed free retention, hinge
deflection and the independent intersection observer.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from stack_grasp import clip_plane,ply_faces


def clip_halfspace(polygon,normal,offset):
    """Keep normal.dot(p) <= offset."""
    if len(polygon)<3:return np.empty((0,3))
    output=[]
    previous=polygon[-1];old=float(previous@normal-offset)
    for current in polygon:
        new=float(current@normal-offset)
        if (old<=0)!=(new<=0):
            output.append(previous+old/(old-new)*(current-previous))
        if new<=0:output.append(current)
        previous,old=current,new
    return np.asarray(output)


def merge_intervals(intervals):
    out=[]
    for low,high in sorted(intervals):
        if high<=low:continue
        if out and low<=out[-1][1]+1e-8:out[-1][1]=max(out[-1][1],high)
        else:out.append([float(low),float(high)])
    return out


def capture_groups(groups,lower,upper,tolerance=.0003):
    """Area and common projected width of four material ply groups."""
    lower,upper=np.asarray(lower),np.asarray(upper)
    r0,r1=Rotation.from_quat(lower[3:]),Rotation.from_quat(upper[3:])
    normal=r0.inv().apply(r1.apply([0,0,1]))
    upper_origin=r0.inv().apply(upper[:3]-lower[:3])
    offset=.0002+tolerance+normal@upper_origin
    reports=[]
    for triangles in groups:
        local=r0.inv().apply((np.asarray(triangles)-lower[:3]).reshape(-1,3)).reshape(-1,3,3)
        area=0.;vertices=[];intervals=[]
        # Broad phase avoids clipping the filled body and remote neck.
        keep=(local[:,:,0].max(1)>0)&(local[:,:,0].min(1)<.044)
        keep&=(local[:,:,1].max(1)>-.05)&(local[:,:,1].min(1)<.05)
        for triangle in local[keep]:
            polygon=triangle
            for axis,bound,greater in ((0,0.,True),(0,.044,False),
                                       (1,-.05,True),(1,.05,False),
                                       (2,-.0002-tolerance,True)):
                polygon=clip_plane(polygon,axis,bound,greater)
                if len(polygon)<3:break
            polygon=clip_halfspace(polygon,normal,offset)
            if len(polygon)<3:continue
            vertices.extend(polygon)
            intervals.append((polygon[:,1].min(),polygon[:,1].max()))
            for i in range(1,len(polygon)-1):
                area+=np.linalg.norm(np.cross(polygon[i]-polygon[0],polygon[i+1]-polygon[0]))/2
        y=np.asarray(vertices)[:,1] if vertices else np.array([])
        reports.append(dict(area_mm2=float(area*1e6),
                            coverage_intervals_m=merge_intervals(intervals),
                            y_min_m=float(y.min()) if len(y) else None,
                            y_max_m=float(y.max()) if len(y) else None))
    width=0.
    if all(r["area_mm2"]>.5 for r in reports):
        width=max(0.,min(r["y_max_m"] for r in reports)-max(r["y_min_m"] for r in reports))
    shared=[[-.05,.05]]
    for report in reports:
        shared=merge_intervals([(max(a,c),min(b,d)) for a,b in shared
                               for c,d in report["coverage_intervals_m"]])
    coverage=sum(b-a for a,b in shared) if all(r["area_mm2"]>.5 for r in reports) else 0.
    return dict(plies=reports,common_span_m=width,common_coverage_m=coverage,
                common_intervals_m=shared,contact_tolerance_m=tolerance,
                limitation="geometry between jaw planes; neither air leakage nor contact load")


def clip_capture(q,faces,grid,lower,upper):
    return capture_groups([q[f] for f in ply_faces(faces,grid)],lower,upper)
