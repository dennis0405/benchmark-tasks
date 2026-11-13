"""Plan a four-ply pinch using film TRIANGLES, including between vertices."""
import numpy as np


def ply_faces(faces,grid):
    nx,ny=grid.shape[1]-1,grid.shape[2]-1
    cells=np.asarray(faces).reshape(2,nx,ny,2,3)
    return [cells[layer,rows].reshape(-1,3) for layer in range(2)
            for rows in (slice(30,34),slice(35,40))]


def clip_plane(polygon,axis,bound,greater):
    output=[]
    previous=polygon[-1]
    was_inside=previous[axis]>=bound if greater else previous[axis]<=bound
    for current in polygon:
        inside=current[axis]>=bound if greater else current[axis]<=bound
        if inside!=was_inside:
            t=(bound-previous[axis])/(current[axis]-previous[axis])
            output.append(previous+t*(current-previous))
        if inside:output.append(current)
        previous,was_inside=current,inside
    return np.asarray(output)


def clipped_surfaces(triangles,bounds,project_xy=False):
    """Total clipped surface area and its vertices; no nodes are moved."""
    if not len(triangles):return 0.,[]
    keep=np.ones(len(triangles),bool)
    for axis,low,high in bounds:
        keep &= (triangles[:,:,axis].max(1)>low)&(triangles[:,:,axis].min(1)<high)
    area=0.;vertices=[]
    for triangle in triangles[keep]:
        polygon=triangle.copy()
        for axis,low,high in bounds:
            polygon=clip_plane(polygon,axis,low,True)
            if len(polygon)<3:break
            polygon=clip_plane(polygon,axis,high,False)
            if len(polygon)<3:break
        if len(polygon)<3:continue
        for i in range(1,len(polygon)-1):
            cross=np.cross(polygon[i]-polygon[0],polygon[i+1]-polygon[0])
            area+=(abs(cross[2]) if project_xy else np.linalg.norm(cross))*.5
        vertices.extend(polygon)
    return float(area),vertices


def select_stack_grasp(points,grid,faces,board_half_width=.06):
    q=np.asarray(points,dtype=float)
    if not np.isfinite(q).all():return None
    groups=[q[f] for f in ply_faces(faces,grid)]
    near=q[q[:,1]>board_half_width+.005]
    if not len(near):return None
    best=None
    for x in np.linspace(near[:,0].min(),near[:,0].max(),81):
        bounds=[(0,x-.0055,x+.0055),(1,board_half_width+.005,board_half_width+.085)]
        reports=[clipped_surfaces(triangles,bounds,True) for triangles in groups]
        areas=[r[0]*1e6 for r in reports]
        if min(areas)<1.:continue
        clipped=np.asarray([v for _,points in reports for v in points])
        z0,z1=float(clipped[:,2].min()),float(clipped[:,2].max())
        if z1-z0>.030:continue
        score=(min(areas),-(z1-z0),sum(areas))
        row=dict(point=[float(x),board_half_width+.045,(z0+z1)/2],
                 ply_projected_area_mm2=areas,z_span_m=z1-z0)
        if best is None or score>best[0]:best=(score,row)
    return best[1] if best else None


def select_press_patch(points,grid,faces,y_center,support_front):
    """Place the 12 x 80 mm pressing pad over the four actual ply groups."""
    q=np.asarray(points,dtype=float)
    groups=[q[f] for f in ply_faces(faces,grid)]
    crease=float(np.median(q[grid[:,34,:],0]))
    low=max(support_front+.0065,crease-.006)
    high=max(low,crease+.030)
    best=None
    for x in np.linspace(low,high,81):
        bounds=[(0,x-.006,x+.006),(1,y_center-.04,y_center+.04)]
        areas=[clipped_surfaces(triangles,bounds,True)[0]*1e6 for triangles in groups]
        score=(min(areas),sum(areas),-abs(x-crease-.008))
        if best is None or score>best[0]:best=(score,dict(x=float(x),ply_area_mm2=areas))
    plan=best[1]
    plan["kind"]="four-ply projected overlap"
    if min(plan["ply_area_mm2"])<.5:
        rim=q[grid[:,-1]].reshape(-1,3)
        rim=rim[abs(rim[:,1]-y_center)<.04]
        if len(rim):
            plan["x"]=float(np.clip(np.median(rim[:,0])+.003,low,high))
            plan["kind"]="measured upright rim fallback"
    return plan


def captured_plies(c,arm="left"):
    entity=c.sim.scene.by_name["pouch"]
    q=c.sim.state.particle_q.numpy()[entity.particles]
    from scipy.spatial.transform import Rotation
    centers,_,_=c.takes._fold_pads(c.film,arm)
    rotation=Rotation.from_quat(c.session.arms[arm].ik.ee_pose(c.sim.state)[1])
    origin=centers.mean(axis=0)
    local=rotation.inv().apply(q-origin)
    center_local=rotation.inv().apply(centers-origin)
    lower,upper=np.sort(center_local[:,1])
    bounds=[(0,-.006,.006),(2,-.04,.04),(1,lower+.002,upper-.002)]
    if lower+.002>=upper-.002:return [0.]*4
    return [clipped_surfaces(local[f],bounds)[0]*1e6 for f in ply_faces(entity.faces,entity.grid)]
