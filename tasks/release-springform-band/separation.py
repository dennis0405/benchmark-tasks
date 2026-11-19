"""Geometry of the forty physical rolled lip facets, measured in the plate frame."""
import numpy as np
from scipy.spatial.transform import Rotation

ANGLE = np.deg2rad(np.arange(4)[:,None]*90 + (np.arange(10)-4.5)*9)
NORMAL = np.stack((np.cos(ANGLE),np.sin(ANGLE),np.zeros_like(ANGLE)),axis=-1)
TANGENT = np.stack((-np.sin(ANGLE),np.cos(ANGLE),np.zeros_like(ANGLE)),axis=-1)
UP = np.array([0.,0.,1.])
# Actual pan.py collider corners. Work in SI; convert reported margins to millimetres.
HALF_LENGTH = .0763*np.tan(np.deg2rad(4.5))+.00005
LIP = np.stack([radius*NORMAL + side*HALF_LENGTH*TANGENT + .005*UP
                for radius in (.074,.0786) for side in (-1,1)],axis=2)

def measure(poses):
    base=poses['base']
    base_axes=Rotation.from_quat(base[3:]).as_matrix()
    cake=poses['cake']
    cake_axes=Rotation.from_quat(cake[3:]).as_matrix()
    sectors=np.asarray([poses[f'sector{i}'] for i in range(4)])
    axes=Rotation.from_quat(sectors[:,3:]).as_matrix()
    world=np.einsum('sij,sfcj->sfci',axes,LIP)+sectors[:,:3][:,None,None,:]
    local=(world-base[:3])@base_axes
    # The rolled lip must clear the plate's upper surface, 1 mm above its origin.
    z=local[:,:,:,2].min(axis=2)-.001
    normals=np.einsum('sij,sfj->sfi',axes,NORMAL)@base_axes
    plane=(local[:,:,0,:]*normals).sum(axis=-1)
    plate_support=.075*np.linalg.norm(normals[:,:,:2],axis=-1)+.001*abs(normals[:,:,2])
    clearance=plane-plate_support
    cake_z=((world-cake[:3])@cake_axes)[:,:,:,2].min()-.024
    return dict(lip_above_plate_mm=(z.ravel()*1000).tolist(),
                lip_radial_margin_mm=(clearance.ravel()*1000).tolist(),lip_above_cake_mm=float(1000*cake_z))
