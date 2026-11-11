"""Read clip-to-packet normal forces from PhysX at each control step."""
import numpy as np


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value.numpy())


class ContactMonitor:
    """Both clip bodies must exert a measurable normal force on a bag body.

    Readings are sampled on the last physics substep of each control step;
    they are simulator estimates, not a real-world force calibration.
    """
    def __init__(self):
        import carb
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_physx.physics.physx_manager import PhysxManager
        from pxr import UsdPhysics

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        sensors, bag_bodies = [], []
        for prim in get_current_stage().Traverse():
            path = prim.GetPath().pathString
            if "/env_0/" not in path:
                continue
            if any(f"/{tool}/" in path for tool in ("clip", "peg")) and \
                    "PhysxContactReportAPI" in prim.GetAppliedSchemas():
                sensors.append(path)
            if "/bag/" in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bag_bodies.append(path)
        self.indices = {tool: [i for i, path in enumerate(sensors) if f"/{tool}/" in path]
                        for tool in ("clip", "peg")}
        if len(bag_bodies) != 3 or any(len(v) != 2 for v in self.indices.values()):
            raise RuntimeError(f"unexpected contact bodies: {self.indices}, bag={bag_bodies}")
        self.view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            sensors, filter_patterns=[bag_bodies] * len(sensors), max_contact_data_count=1024)
        self.physics_dt = PhysxManager.get_physics_dt()

    def read(self):
        forces, _, _, separation, counts, starts = self.view.get_contact_data(dt=self.physics_dt)
        forces, separation, counts, starts = map(_numpy, (forces, separation, counts, starts))
        forces, separation = forces.reshape(-1), separation.reshape(-1)
        totals = np.zeros(counts.shape[0])
        for sensor in range(counts.shape[0]):
            for count, start in zip(counts[sensor], starts[sensor]):
                indices = np.arange(int(start), int(start + count))
                if indices.size:
                    totals[sensor] += np.maximum(forces[indices][separation[indices] <= 0.0002], 0).sum()
        return {tool: all(totals[index] >= 0.001 for index in indices)
                for tool, indices in self.indices.items()}
