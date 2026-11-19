"""Eight independent interference-fit lip sections and a dry, weighted can fixture."""
import math
from dataclasses import replace
import numpy as np
import paint as P
from simkit import Part, Piece, Spring

RADIUS = .068
SEATED_TOP = P.H_CAN + .0015
FREE_TOP = SEATED_TOP + P.PROUD
THICKNESS = .003
CENTER_Z = FREE_TOP - THICKNESS / 2
LUG = np.array([P.CAN_C[0], P.CAN_C[1]+.147, P.FLOOR+.110])
SECTORS = tuple(f'rim{i}' for i in range(8))


def can():
    original = P.can_object()
    # Fixed dry ballast replaces paint. A side tab on the stabilizing collar can
    # be pinched from above without obstructing the hammer. Centre disk is recessed.
    color = (.18, .38, .43)
    extra = (Piece('box', (.014,.076,.018), (0,.121,.110-P.H_CAN/2),color=color),
             Piece('cylinder', (.054,.003,.054), (0,0,SEATED_TOP-.004-P.H_CAN/2),color=P.TIN))
    meshes = tuple(m for m in original.vmeshes if 'paint' not in m[0] and not m[0].startswith(('drip','drop')))
    return replace(original, pieces=original.pieces+extra,vmeshes=meshes)


def sector(i):
    a=math.tau*i/8
    offset=np.array([RADIUS*math.cos(a),RADIUS*math.sin(a)])
    # Three small staves approximate one annular sector; all are on the same physical slide.
    staves=[]
    for da in (-.255, 0., .255):
        angle=a+da
        delta=RADIUS*np.array([math.cos(angle),math.sin(angle)])-offset
        staves.append(Piece('box',(.025,.0115,THICKNESS),(*delta,0.),rot=(0.,0.,angle),
                            color=(.68,.70,.72),hidden=True))
    pos=tuple(map(float, (*np.add(P.CAN_C,offset),P.FLOOR+CENTER_Z)))
    # Continuous annular lip surface over the three collision patches.
    profile=[(.052,-.002),(.060,-.0008),(.070,-.0004),(.075,.0004),
             (.078,.0014),(.080,.0014),(.080,-.0014)]
    mesh=P.M.revolve(profile,n=16,a0=a-math.pi/8+.002,a1=a+math.pi/8-.002)
    vis=P._vm('formed_lip',mesh,(.66,.68,.71),offset=(-offset[0],-offset[1],0.))
    return Part(SECTORS[i],pos=pos,size=(.003,.003,.002),exact_z=True,mass=.00875,
                color=P.TIN,pieces=tuple(staves),vmeshes=(vis,),hide_root=True,contact_offset=.00015,
                spring=Spring(axis='Z', lower=-P.PROUD,upper=.0001,stiffness=4000.,damping=2.,
                              target=0.,parent='can',parent_pos=P.CAN_POS))


def tool():
    # Face-down pickup puts the wrist above the head. A saddle supports the handle at rest.
    original=P.mallet_object()
    turn=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    meshes=tuple((v[0],np.asarray(v[1])@turn.T,*v[2:]) for v in original.vmeshes)
    return replace(original,axis='Z',pos=(*P.HEAD_C,P.FLOOR+P.HEAD_L/2),vmeshes=meshes)


def handle_rest():
    where=np.array(P.HEAD_C)+P.mallet_dir()[:2]*.195
    height=P.HEAD_L/2-.012
    return Part('rest',pos=(*map(float,where),P.FLOOR+height/2),size=(.038,.04,height),
                color=(.26,.37,.40),exact_z=True,friction=(.72,.65))
