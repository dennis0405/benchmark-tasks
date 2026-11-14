"""Opt-in, pinned Newton 1.7 contact reduction specialization.

Only the number of strided threads per rigid body changes. Contacts,
iterations, force laws and independent geometry checks are unchanged.
Call before any VBD scene is built in this process. No installed file is edited.
"""
import hashlib
from pathlib import Path
import warp as wp


HASHES={"rigid_vbd_kernels":"8f0dc0fd50477025f461a9398f4a51f7dd8a499ae806230be82bc7e00fd1a412",
        "solver_vbd":"52d9de6f8cc662a95b4c55eaa657c3eedf5f6a6da699109dd9c2de36e36b932f"}

_ACTIVE_WIDTH=None

def configure(threads):
    global _ACTIVE_WIDTH
    if _ACTIVE_WIDTH is not None and threads!=_ACTIVE_WIDTH:
        raise RuntimeError("Use a fresh process to change compiled VBD contact thread width")
    _ACTIVE_WIDTH=threads
    if threads not in (4,8,16,32,64):raise ValueError("supported contact thread widths: 4,8,16,32,64")
    if threads==4:return dict(threads=4,specialization=False)
    from newton._src.solvers.vbd import rigid_vbd_kernels,solver_vbd
    for name,module in (("rigid_vbd_kernels",rigid_vbd_kernels),("solver_vbd",solver_vbd)):
        digest=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest!=HASHES[name]:raise RuntimeError(f"Unverified Newton source version for {name}")
    value=wp.constant(threads)
    rigid_vbd_kernels._NUM_CONTACT_THREADS_PER_BODY=value
    solver_vbd._NUM_CONTACT_THREADS_PER_BODY=value
    return dict(threads=threads,specialization=True,source_sha256=HASHES)
