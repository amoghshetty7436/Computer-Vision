"""Pick the Euler convention for Boreas camera_poses.csv (roll, pitch, heading) that best
agrees with a clean daytime COLMAP reconstruction. Relative-rotation *magnitudes* do not
depend on the convention, so they validate that SfM itself is accurate; full relative
rotations then select the convention."""
import itertools, sys, os
import numpy as np
from pathlib import Path
from sfmrob import data as D
from sfmrob.sfm import run_sfm, SfMConfig
from sfmrob.metrics import poses_from_reconstruction, _rotmat_angle

seq = "boreas-2021-04-08-12-44"
P = D.boreas_camera_poses(seq)
i = 3000
idx = D.boreas_select(seq, P[i, 1:3], np.arctan2(P[i, 5], P[i, 4]))
work = Path(os.environ.get("SFMROB_WORK", "/tmp/sfmrob_work")) / "boreas_check"
names, ref, K = D.boreas_fetch(seq, idx, 1024, work / "images")
cfg = SfMConfig(camera_params=f"{K['f']},{K['cx']},{K['cy']},0")
res = run_sfm(work / "images", work / "sfm", cfg, image_names=names)
est = poses_from_reconstruction(res.best)
name2i = {n: j for n, j in zip(names, idx)}
common = [n for n in names if n in est]
print("registered", len(common), "/", len(names))
R = D._rot
best = None
for order in itertools.permutations("xyz"):
    for transpose in [False, True]:
        errs = []
        def Rwc(n):
            r, p, h = P[name2i[n], 7], P[name2i[n], 8], P[name2i[n], 9]
            ang = dict(x=r, y=p, z=h)
            M = R(order[0], ang[order[0]]) @ R(order[1], ang[order[1]]) @ R(order[2], ang[order[2]])
            return M.T if transpose else M
        for a, b in zip(common[:-5], common[5:]):
            Ra, Rb = Rwc(a).T, Rwc(b).T  # cam_from_world
            rel_ref = Rb @ Ra.T
            rel_est = est[b][0] @ est[a][0].T
            errs.append(_rotmat_angle(rel_est.T @ rel_ref))
        e = np.median(errs)
        print("".join(order), "T" if transpose else " ", f"median rel-rot err {e:.3f} deg")
        if best is None or e < best[0]:
            best = (e, order, transpose)
print("BEST", best)
