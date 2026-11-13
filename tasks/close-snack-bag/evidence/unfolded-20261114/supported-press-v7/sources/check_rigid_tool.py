"""Independent clip BOX / BOX preflight, including static fixtures.

This does not inspect robot capsules or mesh colliders. It includes opposite
clip parts even when the rigid solver filters their adjacent-body contacts.
"""
import numpy as np
import newton
import warp as wp
from gpu_contact_checks import rigid_boxes


def inspect_tool(env):
    model=env.model
    tool=env.scene.by_name["clip"]
    owners=model.shape_body.numpy().copy()
    kinds=model.shape_type.numpy()
    flags=model.shape_flags.numpy()
    eligible=np.flatnonzero((kinds==int(newton.GeoType.BOX)) &
        ((flags&int(newton.ShapeFlags.COLLIDE_SHAPES))!=0))
    selected=set(tool.shapes)&set(map(int,eligible))
    pairs=sorted({tuple(sorted((int(a),int(b)))) for a in selected for b in eligible
                  if a!=b and owners[a]!=owners[b]})
    body=env.state.body_q.numpy()
    owners[owners<0]=len(body)
    body=np.vstack([body,[0,0,0,0,0,0,1]])
    counts=wp.zeros(1,dtype=int,device="cpu")
    hit=wp.zeros(len(pairs),dtype=int,device="cpu")
    if pairs:
        wp.launch(rigid_boxes,len(pairs),inputs=[wp.array(body,dtype=wp.transform,device="cpu"),
            wp.array(owners,dtype=int,device="cpu"),wp.array(model.shape_transform.numpy(),dtype=wp.transform,device="cpu"),
            wp.array(model.shape_scale.numpy(),dtype=wp.vec3,device="cpu"),
            wp.array(np.asarray(pairs),dtype=wp.vec2i,device="cpu"),0,counts,hit],device="cpu")
    return dict(scope=__doc__.strip(),checked_pairs=len(pairs),
                intersections=[dict(shapes=list(pairs[i]),
                    labels=[str(model.shape_label[s]) for s in pairs[i]]) for i in np.flatnonzero(hit.numpy())])
