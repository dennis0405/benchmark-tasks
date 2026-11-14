"""Measured, irreversible failure of pre-fragmented cracker seams.

The constitutive thresholds are uncalibrated. GPU observation covers each
physical substep; disabling a failed seam occurs at the next control boundary
so VBD contact conditioning can be refreshed outside a captured graph.
"""
import numpy as np
import newton
import warp as wp


@wp.kernel
def observe_seams(poses:wp.array[wp.transform],pairs:wp.array[wp.vec2i],
                  joints:wp.array[int],enabled:wp.array[bool],counter:wp.array[int],
                  threshold:float,gap_threshold:float,peak_angle:wp.array[float],
                  peak_gap:wp.array[float],pending:wp.array[int],first_hit:wp.array[int]):
    i=wp.tid();pair=pairs[i]
    if enabled[joints[i]]:
        a=poses[pair[0]];b=poses[pair[1]]
        relative=wp.quat_inverse(wp.transform_get_rotation(a))*wp.transform_get_rotation(b)
        angle=2.*wp.acos(wp.clamp(wp.abs(relative[3]),0.,1.))
        p=wp.transform_point(a,wp.vec3(.012,0.,0.))
        q=wp.transform_point(b,wp.vec3(-.012,0.,0.))
        gap=wp.length(p-q)
        peak_angle[i]=wp.max(peak_angle[i],angle);peak_gap[i]=wp.max(peak_gap[i],gap)
        if angle>threshold or gap>gap_threshold:
            pending[i]=1
            if first_hit[i]<0:first_hit[i]=counter[0]


@wp.kernel
def increment(counter:wp.array[int]):counter[0]+=1


class BrittleSeams:
    def __init__(self,sim,*,initial_peaks=None):
        self.sim=sim;entity=sim.scene.by_name["contents"]
        self.joints=np.asarray(entity.seams,dtype=np.int32)
        if len(self.joints)!=len(entity.chip_groups):raise ValueError("each cracker needs one seam")
        device=sim.model.device;n=len(self.joints)
        self.pairs=wp.array(entity.chip_groups,dtype=wp.vec2i,device=device)
        self.indices=wp.array(self.joints,dtype=int,device=device)
        self.peak_angle=wp.zeros(n,dtype=float,device=device);self.peak_gap=wp.zeros(n,dtype=float,device=device)
        if initial_peaks is not None:self.peak_angle.assign(initial_peaks)
        self.pending=wp.zeros(n,dtype=int,device=device)
        self.first_hit=wp.array(np.full(n,-1,dtype=np.int32),dtype=int,device=device)
        self.counter=wp.zeros(1,dtype=int,device=device)
        self.broken=~sim.model.joint_enabled.numpy()[self.joints].astype(bool)
        self.events=[];self.threshold=.12;self.gap_threshold=.0004

    def observe(self,state,dt):
        del dt
        wp.launch(observe_seams,len(self.joints),inputs=[state.body_q,self.pairs,self.indices,
            self.sim.model.joint_enabled,self.counter,self.threshold,self.gap_threshold,
            self.peak_angle,self.peak_gap,self.pending,self.first_hit],device=self.sim.model.device)
        wp.launch(increment,1,inputs=[self.counter],device=self.sim.model.device)

    def commit(self,step):
        newly=(self.pending.numpy()!=0)&~self.broken
        if not newly.any():return
        enabled=self.sim.model.joint_enabled.numpy().copy()
        enabled[self.joints[newly]]=False;self.sim.model.joint_enabled.assign(enabled)
        self.sim.pipeline.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        self.sim.pipeline.invalidate_graph()
        first=self.first_hit.numpy();angles=self.peak_angle.numpy();gaps=self.peak_gap.numpy()
        for chip in np.flatnonzero(newly):
            self.events.append(dict(chip=int(chip),control_step=int(step),
                first_detected_physics_sample=int(first[chip]),peak_bend_rad=float(angles[chip]),
                peak_anchor_gap_m=float(gaps[chip])))
        self.broken|=newly

    def report(self):
        return dict(broken_count=int(self.broken.sum()),broken=self.broken.tolist(),
            peak_bend_rad=self.peak_angle.numpy().tolist(),peak_anchor_gap_m=self.peak_gap.numpy().tolist(),
            observed_substeps=int(self.counter.numpy()[0]),threshold_rad=self.threshold,
            gap_threshold_m=self.gap_threshold,calibrated=False,
            event_latency_bound_s=1./float(self.sim.request.physics.control_hz))
