"""Per-image pose diagnosis of one reconstruction vs the reference model.

  python scripts/diag_poses.py <scene> <sparse_dir>

Prints, for each registered image (in name order), the median relative-rotation and
translation-direction error against all other registered images. A contiguous block of
images with large, similar errors is the signature of a symmetry flip (see
docs/notes_glomap_symmetry.md); uniformly large errors point to a calibration problem."""
import sys

import numpy as np
import pycolmap

from sfmrob import data as D
from sfmrob.metrics import _rotmat_angle, _vec_angle, poses_from_reconstruction

scene, sparse = sys.argv[1], sys.argv[2]
rec = pycolmap.Reconstruction(sparse)
for c in rec.cameras.values():
    print("camera", c.model, c.width, c.height, np.round(c.params, 3))
est = poses_from_reconstruction(rec)
ref = D.load_colmap_reference(D.fetch_colmap_scene(scene))
n = sorted(k for k in est if k in ref)
R = np.zeros((len(n), len(n))); T = np.zeros_like(R)
for i, a in enumerate(n):
    for j, b in enumerate(n):
        if i == j:
            continue
        Ra, ta = ref[a]; Rb, tb = ref[b]; Rr = Rb @ Ra.T; tr = tb - Rr @ ta
        Ea, ea = est[a]; Eb, eb = est[b]; Re = Eb @ Ea.T; te = eb - Re @ ea
        R[i, j] = _rotmat_angle(Re.T @ Rr); T[i, j] = _vec_angle(te, tr)
med_t = np.nanmedian(np.where(np.eye(len(n)) > 0, np.nan, T), axis=1)
med_r = np.nanmedian(np.where(np.eye(len(n)) > 0, np.nan, R), axis=1)
for k, (a, r, t) in enumerate(zip(n, med_r, med_t)):
    print(f"{k:3d} {a:20s} rot {r:7.2f}  trans {t:7.2f}")
print("images with median rot err > 10 deg:", int((med_r > 10).sum()), "of", len(n))
