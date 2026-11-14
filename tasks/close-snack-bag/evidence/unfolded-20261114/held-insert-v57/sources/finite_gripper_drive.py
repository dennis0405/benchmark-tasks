"""Force-limited opposed-finger drive for the unified VBD experiment.

This is an explicit compliant transmission approximation to the imported
mimic gripper. Both slides receive forces; neither poses nor joint coordinates
are projected. Available actuator force is 10 N per hand; passive transmission
reactions arise from a centering spring and damper.
"""
import numpy as np
import warp as wp


def configure(model):
    if hasattr(model,"snack_gripper_drives"):return
    if not model.constraint_mimic_count:
        model.snack_gripper_drives=[]
        return
    rows=[]
    ke=model.joint_target_ke.numpy().copy();kd=model.joint_target_kd.numpy().copy()
    starts=model.joint_qd_start.numpy();qstarts=model.joint_q_start.numpy()
    for j0,j1,a,b,on in zip(model.constraint_mimic_joint0.numpy(),model.constraint_mimic_joint1.numpy(),
            model.constraint_mimic_coef0.numpy(),model.constraint_mimic_coef1.numpy(),model.constraint_mimic_enabled.numpy()):
        if not on:continue
        j0,j1=int(j0),int(j1)
        if ke[starts[j0]]>0 and ke[starts[j1]]==0:
            reference,follower=j0,j1;offset,multiplier=-float(a)/float(b),1./float(b)
        elif ke[starts[j1]]>0 and ke[starts[j0]]==0:
            reference,follower=j1,j0;offset,multiplier=float(a),float(b)
        else:raise ValueError("Expected one driven and one passive slide")
        for joint,off,mul in ((reference,0.,1.),(follower,offset,multiplier)):
            dof=int(starts[joint]);coord=int(qstarts[joint])
            rows.append((dof,coord,int(qstarts[reference]),off,mul))
            ke[dof]=0.;kd[dof]=0.
    model.joint_target_ke.assign(ke);model.joint_target_kd.assign(kd)
    model.snack_gripper_drives=rows


@wp.kernel
def drive(q:wp.array[float],qd:wp.array[float],target:wp.array[float],
          indices:wp.array[wp.vec3i],coeff:wp.array[wp.vec2],
          stiffness:float,damping:float,limit:float,force:wp.array[float]):
    i=wp.tid();index=indices[i];c=coeff[i]
    goal=c[0]+c[1]*target[index[2]]
    force[index[0]]=wp.clamp(stiffness*(goal-q[index[1]])-damping*qd[index[0]],-limit,limit)


@wp.kernel
def geared_drive(q:wp.array[float],qd:wp.array[float],target:wp.array[float],
                 pairs:wp.array[wp.vec4i],target_index:wp.array[int],
                 force:wp.array[float],observed:wp.array[wp.vec2]):
    i=wp.tid();p=pairs[i]
    opening=0.5*(q[p[1]]-q[p[3]])
    speed=0.5*(qd[p[0]]-qd[p[2]])
    actuator=wp.clamp(10000.0*(target[target_index[i]]-opening)-10.0*speed,-5.0,5.0)
    # A passive compliant centering transmission. Its stored energy is
    # 0.5*k*(q0+q1)^2, and damping dissipates d*(qd0+qd1)^2.
    reaction=-25000.0*(q[p[1]]+q[p[3]])-10.0*(qd[p[0]]+qd[p[2]])
    force[p[0]]=actuator+reaction
    force[p[2]]=-actuator+reaction
    observed[i]=wp.vec2(actuator,reaction)


class FiniteGripperDrive:
    def __init__(self,sim):
        self.sim=sim;rows=sim.model.snack_gripper_drives
        pairs=[];targets=[]
        for leader,follower in zip(rows[::2],rows[1::2]):
            if leader[3:]!=(0.,1.) or follower[3:]!=(0.,-1.):
                raise ValueError("Geared drive currently requires opposed, zero-offset YAM slides")
            pairs.append((leader[0],leader[1],follower[0],follower[1]));targets.append(leader[2])
        self.pairs=wp.array(pairs,dtype=wp.vec4i,device=sim.model.device)
        self.targets=wp.array(targets,dtype=int,device=sim.model.device)
        self.observed=wp.zeros(len(pairs),dtype=wp.vec2,device=sim.model.device)
        self.metadata=dict(model="force-limited opposed drive with passive compliant centering transmission",
                           stiffness_N_m=10000.,damping_N_s_m=10.,actuator_force_N_per_slide=5.,
                           transmission_stiffness_N_m=25000.,transmission_damping_N_s_m=10.,rows=rows)

    def before_solve(self,state,dt):
        del dt
        wp.launch(geared_drive,len(self.pairs),inputs=[state.joint_q,state.joint_qd,self.sim.control.joint_target_q,
                  self.pairs,self.targets],outputs=[self.sim.control.joint_f,self.observed],device=self.sim.model.device)
