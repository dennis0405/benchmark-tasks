# Recorded folding attempt for visual review

`snack-bag-fold-review.mp4` is a 9.35 s, 20 fps, 1280 x 640 diagnostic video
of the uninterrupted `edge-grasp-seal-v58` episode. It shows the unfolded
reset, mouth alignment, pinching and attempted fold. The fold handoff fails;
clip insertion, release and hands-free retention are never reached. Four of
twelve cracker seams are flagged broken by the uncalibrated damage model.

The left view shows the full robot. The right view makes the recorded robot
translucent so the mouth is visible; this is explicitly labelled in-video.
No object position is interpolated or edited. The render manifest hashes the
original complete recording, trace, renderer and MP4. MP4 metadata was checked
and an encoded frame was decoded for visual inspection.

`execution-source.tar.gz` contains the exact isolated execution tree, verified
against its manifest; `render-source.py` is the separately hashed renderer.
Only initial/final states are included locally. Complete raw states remain on
Brev under `/data/robocurve/runs/edge-grasp-seal-v58`.

At the user's request for review before further changes, v65 and v66 are
saved and explicitly interrupted at 139 and 90 frames. They are incomplete,
not failures of their final gates or complete comparisons. No further trial
is running; the Brev instance is left on awaiting the user's guidance.
