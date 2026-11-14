"""Isolated known-sum check for the optional VBD reduction specialization."""
import argparse,json,time
import numpy as np
import warp as wp
from vbd_parallel import configure
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=4)
p.add_argument('--contacts',type=int,default=100);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
configure(a.threads)
from newton._src.solvers.vbd.rigid_vbd_kernels import accumulate_body_particle_contacts_per_body
n=a.contacts;device='cuda:0'
def arr(x,dtype):return wp.array(x,dtype=dtype,device=device)
def zeros(dtype):return wp.zeros(1,dtype=dtype,device=device)
q=arr([[0,0,.0002]]*n,wp.vec3);body=arr([[0,0,0,0,0,0,1]],wp.transform)
v=arr([[0,0,0]]*n,wp.vec3);counts=arr([n],int);cap=max(128,n)
force,torque,hll,hal,haa=zeros(wp.vec3),zeros(wp.vec3),zeros(wp.mat33),zeros(wp.mat33),zeros(wp.mat33)
inputs=[.001,arr([0],int),q,q,arr([.0003]*n,float),body,body,zeros(wp.spatial_vector),
        zeros(wp.vec3),arr([1],float),arr([0],int),.001,False,
        arr([10000]*n,float),arr([10000]*n,float),arr([0]*n,float),arr([0]*n,float),
        counts,arr([[i,-1,-1] for i in range(n)],wp.vec3i),arr([0]*n,int),v,v,
        arr([[0,0,1]]*n,wp.vec3),arr([[1,0,0]]*n,wp.vec3),arr([0],float),cap,counts,
        arr(list(range(n))+[0]*(cap-n),int)]
wp.launch(accumulate_body_particle_contacts_per_body,a.threads,inputs=inputs,
          outputs=[force,torque,hll,hal,haa],device=device)
actual=force.numpy()[0]
np.testing.assert_allclose(actual,[0,0,-n],atol=max(.001,n*.0001))
timings=[]
if a.repeat:
    with wp.ScopedCapture(device=device) as capture:
        for value in (force,torque,hll,hal,haa):value.zero_()
        wp.launch(accumulate_body_particle_contacts_per_body,a.threads,inputs=inputs,
                  outputs=[force,torque,hll,hal,haa],device=device)
    for _ in range(5):
        wp.synchronize_device(device);started=time.perf_counter()
        for __ in range(a.repeat):wp.capture_launch(capture.graph)
        wp.synchronize_device(device);timings.append((time.perf_counter()-started)/a.repeat)
print(json.dumps(dict(threads=a.threads,contacts=n,force=actual.tolist(),expected=[0,0,-n],
    per_graph_s=timings,median_graph_s=float(np.median(timings)) if timings else None)))
