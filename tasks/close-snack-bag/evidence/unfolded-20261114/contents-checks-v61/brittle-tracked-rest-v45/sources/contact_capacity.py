"""Check rigid and rigid/film contact capacities after every physical substep.

Read-only. Reports dropped-contact risk instead of treating a finite solve
with an overflowing adjacency list as a validated contact trajectory.
"""
import warp as wp


@wp.kernel
def observe(bodies:int,rigid_counts:wp.array[int],soft_counts:wp.array[int],
            rigid_capacity:int,soft_capacity:int,rigid_total:wp.array[int],soft_total:wp.array[int],
            rigid_max:int,soft_max:int,stats:wp.array[int]):
    i=wp.tid()
    if i<bodies:
        r=rigid_counts[i];s=soft_counts[i]
        wp.atomic_max(stats,1,r);wp.atomic_max(stats,2,s)
        if r>rigid_capacity:wp.atomic_add(stats,3,1)
        if s>soft_capacity:wp.atomic_add(stats,4,1)
    if i==0:
        wp.atomic_add(stats,0,1)
        if rigid_total[0]>rigid_max:wp.atomic_add(stats,5,1)
        if soft_total[0]>soft_max:wp.atomic_add(stats,6,1)


class ContactCapacity:
    def __init__(self,solver,contacts):
        self.solver=solver;self.contacts=contacts
        self.stats=wp.zeros(7,dtype=int,device=solver.model.device)
    def check(self,state,dt):
        del state,dt
        s=self.solver;c=self.contacts
        wp.launch(observe,max(1,s.model.body_count),inputs=[s.model.body_count,s.body_body_contact_counts,
            s.body_particle_contact_counts,int(s.body_body_contact_buffer_pre_alloc),
            int(s.body_particle_contact_buffer_pre_alloc),c.rigid_contact_count,c.soft_contact_count,
            int(c.rigid_contact_max),int(c.soft_contact_max)],outputs=[self.stats],device=s.model.device)
    def report(self):
        names=('checked_substeps','max_rigid_contacts_per_body','max_soft_contacts_per_body',
               'rigid_body_overflow_events','soft_body_overflow_events','rigid_total_overflow_substeps',
               'soft_total_overflow_substeps')
        return dict(zip(names,map(int,self.stats.numpy())))
