"""PhysX break notifications are independent of task state and controller phase."""
class Breaks:
    def __init__(self):
        import omni.physx
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools as PST
        self.names=set()
        def on_event(e):
            if e.type==int(SimulationEvent.JOINT_BREAK):
                path=str(PST.decodeSdfPath(*e.payload['jointPath']))
                self.names.add(path.split('/')[-1])
        self.events=omni.physx.get_physx_interface().get_simulation_event_stream_v2()
        self.sub=self.events.create_subscription_to_pop(on_event)
    def fields(self):
        return dict(inner_broken='InnerTab' in self.names,outer_broken='OuterTab' in self.names)

