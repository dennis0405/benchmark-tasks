"""Experimental implicit gas/film response; opt-in, not integrated in the demo.

The old explicit PV force is stiff relative to a gram-scale wrapper. This
uses an implicit orifice step and a linearized inertial volume response.
It remains an isothermal, lumped leak proxy, not CFD or a measured air seal.
"""
import math
import numpy as np
import warp as wp
from pouch_air import R
from pouch_air_warp import (LumpedGas,_accumulate_geometry,_mouth_geometry,
                            _contents_volume,_init_channel,_channel_area)

RT_AIR=wp.constant(wp.float64(R*293.15))


def implicit_orifice(gauge,volume,area,dt,ambient=101325.,density=1.2,discharge=.6):
    """Solve p + alpha sign(p) sqrt(abs(p)) = p_before exactly."""
    if not all(math.isfinite(v) for v in (gauge,volume,area,dt,ambient,density,discharge)):
        raise ValueError("finite gas values required")
    if min(volume,dt,ambient,density)<=0 or area<0 or not 0<=discharge<=1:
        raise ValueError("invalid gas parameters")
    if not gauge:return 0.
    alpha=ambient/volume*dt*discharge*area*math.sqrt(2/density)
    root=2*abs(gauge)/(math.sqrt(alpha*alpha+4*abs(gauge))+alpha)
    return math.copysign(root*root,gauge)


def inertial_pressure(gauge,volume,volume_rate,inverse_volume_mass,dt,ambient=101325.):
    """Linearize pressure after q += dt*v + dt²*M^-1*grad(V)*p."""
    if min(volume,dt,ambient+gauge)<=0 or inverse_volume_mass<0:
        raise ValueError("invalid volume response")
    bulk=(ambient+gauge)/volume
    return (gauge-bulk*dt*volume_rate)/(1+bulk*dt*dt*inverse_volume_mass)


@wp.kernel
def _gradient(q:wp.array[wp.vec3],faces:wp.array[wp.vec3i],g:wp.array[wp.vec3]):
    t=faces[wp.tid()]
    n=wp.cross(q[t[1]]-q[t[0]],q[t[2]]-q[t[0]])/6.
    for j in range(3):wp.atomic_add(g,t[j],n)


@wp.kernel
def _response(g:wp.array[wp.vec3],v:wp.array[wp.vec3],inverse_mass:wp.array[float],
              response:wp.array[wp.float64]):
    i=wp.tid()
    wp.atomic_add(response,0,wp.float64(wp.dot(g[i],v[i])))
    wp.atomic_add(response,1,wp.float64(wp.dot(g[i],g[i])*inverse_mass[i]))


@wp.kernel
def _implicit_gas(stats:wp.array[float],response:wp.array[wp.float64],
                  moles:wp.array[wp.float64],dt:float,use_channel:int):
    volume=wp.float64(stats[0]-stats[3])
    if volume<=1.e-10 or stats[4]>0.:
        stats[2]=-1.e9
    else:
        ambient=wp.float64(101325.)
        gauge=moles[0]*RT_AIR/volume-ambient
        area=wp.float64(wp.where(use_channel!=0,wp.min(stats[1],stats[5]),stats[1]))
        alpha=ambient/volume*wp.float64(dt*.6)*area*wp.sqrt(wp.float64(2./1.2))
        root=wp.float64(0.)
        if wp.abs(gauge)>0.:
            root=wp.float64(2.)*wp.abs(gauge)/(wp.sqrt(alpha*alpha+wp.float64(4.)*wp.abs(gauge))+alpha)
        vented=wp.sign(gauge)*root*root
        moles[0]=(ambient+vented)*volume/RT_AIR
        bulk=(ambient+vented)/volume
        pressure=(vented-bulk*wp.float64(dt)*response[0])/(wp.float64(1.)+bulk*wp.float64(dt*dt)*response[1])
        response[2]=vented
        stats[2]=wp.float32(pressure)
        if pressure<=-ambient:stats[2]=-1.e9


@wp.kernel
def _force(g:wp.array[wp.vec3],stats:wp.array[float],f:wp.array[wp.vec3]):
    i=wp.tid()
    if stats[2]>-1.e8:wp.atomic_add(f,i,stats[2]*g[i])


class ImplicitPouchGas(LumpedGas):
    def __init__(self,model,packet_points,packet_faces,grid,start,solver_dt,contact_gap=.0006):
        super().__init__(model,packet_points,packet_faces,grid,start,solver_dt,
                         channel=True,contact_gap=contact_gap)
        self.model=model
        self.moles=wp.array(self.moles.numpy().astype(np.float64),dtype=wp.float64,device=self.device)
        self.gradient=wp.zeros(model.particle_count,dtype=wp.vec3,device=self.device)
        self.response=wp.zeros(3,dtype=wp.float64,device=self.device)

    def before_solve(self,state,dt):
        if abs(dt-self.dt)>1e-9:raise ValueError("gas callback dt mismatch")
        self.stats.zero_();self.gradient.zero_();self.response.zero_()
        wp.launch(_accumulate_geometry,len(self.closed),inputs=[state.particle_q,self.closed,self.start,self.stats],device=self.device)
        wp.launch(_mouth_geometry,len(self.cap),inputs=[state.particle_q,self.cap,self.stats],device=self.device)
        wp.launch(_init_channel,1,inputs=[self.stats],device=self.device)
        wp.launch(_channel_area,self.grid.shape[1]-self.first_column,
                  inputs=[state.particle_q,self.grid,self.first_column,self.contact_gap,self.stats],device=self.device)
        if self.tets is not None:
            wp.launch(_contents_volume,len(self.tets),inputs=[state.particle_q,self.tets,self.orientation,self.stats],device=self.device)
        wp.launch(_gradient,len(self.closed),inputs=[state.particle_q,self.closed,self.gradient],device=self.device)
        wp.launch(_response,self.model.particle_count,inputs=[self.gradient,state.particle_qd,self.model.particle_inv_mass,self.response],device=self.device)
        wp.launch(_implicit_gas,1,inputs=[self.stats,self.response,self.moles,dt,1],device=self.device)
        wp.launch(_force,self.model.particle_count,inputs=[self.gradient,self.stats,state.particle_f],device=self.device)

    def report(self):
        stats=self.stats.numpy();response=self.response.numpy()
        return dict(valid=bool(stats[2]>-1e8),cavity_volume_m3=float(stats[0]),
                    air_volume_m3=float(stats[0]-stats[3]),opening_proxy_m2=float(min(stats[1],stats[5])),
                    gauge_after_vent_Pa=float(response[2]),applied_pressure_Pa=float(stats[2]),
                    moles=float(self.moles.numpy()[0]),
                    limitation="Linearized inertial gas-film coupling; virtual cap traction; paired-skin leak proxy; no resolved airflow")
