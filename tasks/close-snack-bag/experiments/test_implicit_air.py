import math
import unittest
from implicit_pouch_air import implicit_orifice,inertial_pressure
from implicit_pouch_air import _gradient,_response,_implicit_gas,_force
import numpy as np
import warp as wp


class ImplicitGasReferenceTests(unittest.TestCase):
    def test_orifice_satisfies_implicit_flow_equation_without_overshoot(self):
        for before in (-1000.,-.01,.01,1000.):
            for dt in (1/600,1/2400,.1):
                after=implicit_orifice(before,1e-4,1e-4,dt)
                alpha=101325/1e-4*dt*.6*1e-4*math.sqrt(2/1.2)
                self.assertAlmostEqual(after+alpha*math.copysign(math.sqrt(abs(after)),after),before)
                self.assertLess(abs(after),abs(before))

    def test_sealed_orifice_does_not_discard_gas(self):
        self.assertAlmostEqual(implicit_orifice(100.,1e-4,0.,.01),100.)

    def test_inertia_predictor_satisfies_linearized_volume_equation(self):
        dt=1/2400;v=1e-4;rate=-.001;w=.8;p0=100.
        p=inertial_pressure(p0,v,rate,w,dt)
        bulk=(101325+p0)/v
        self.assertAlmostEqual(p,p0-bulk*(dt*rate+dt*dt*w*p),places=7)
        self.assertGreater(p,0.)

    def test_warp_bridge_matches_reference_and_conserves_closed_surface_force(self):
        for device in ("cpu","cuda:0") if wp.is_cuda_available() else ("cpu",):
            with self.subTest(device=device):
                q=wp.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],dtype=wp.vec3,device=device)
                faces=wp.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=wp.vec3i,device=device)
                g=wp.zeros(4,dtype=wp.vec3,device=device)
                v=wp.zeros(4,dtype=wp.vec3,device=device)
                mass=wp.array(np.ones(4,np.float32),dtype=float,device=device)
                response=wp.zeros(3,dtype=wp.float64,device=device)
                stats=wp.array([1e-4,1e-4,0,0,0,1e-4],dtype=float,device=device)
                volume=float(stats.numpy()[0]);dt=1/2400
                moles=wp.array([(101325+100)*volume/(8.314462618*293.15)],dtype=wp.float64,device=device)
                wp.launch(_gradient,4,inputs=[q,faces,g],device=device)
                wp.launch(_response,4,inputs=[g,v,mass,response],device=device)
                wp.launch(_implicit_gas,1,inputs=[stats,response,moles,dt,1],device=device)
                wp.launch(_force,4,inputs=[g,stats,v],device=device)
                r=response.numpy();s=stats.numpy()
                vented=implicit_orifice(100.,volume,float(s[1]),dt)
                expected=inertial_pressure(vented,volume,0.,float(r[1]),dt)
                self.assertAlmostEqual(float(s[2]),expected,places=4)
                np.testing.assert_allclose(v.numpy().sum(0),0,atol=1e-7)


if __name__=="__main__":unittest.main()
