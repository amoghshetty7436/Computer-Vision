"""ALIKED + LightGlue (kornia implementations) written into a COLMAP database, followed by
COLMAP's own geometric verification, so that only the *features/matcher* differ from the
SIFT pipeline (same mapper, same RANSAC, same BA).

Weights: ALIKED from the authors' GitHub repo, LightGlue from cvg/LightGlue releases.
Note: LightGlue is ~10 s/pair on CPU and ~20 ms/pair on an A100; exhaustive matching
is therefore meant for the GPU tier (Perlmutter)."""
from __future__ import annotations

import itertools
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image

ALIKED_URL = "https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth"


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(1)
def _models(max_kpts=4096):
    import kornia.feature as KF
    ck = Path(torch.hub.get_dir()) / "checkpoints" / "aliked-n16.pth"
    if not ck.exists():
        ck.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ALIKED_URL, ck)
    ext = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_kpts, device=_dev()).eval()
    lg = KF.LightGlue("aliked").eval().to(_dev())
    return ext, lg


@torch.no_grad()
def _mnn_ratio(da, db, ratio=0.9):
    """Mutual nearest neighbours + Lowe ratio test on L2-normalised descriptors (the classic
    SIFT-style matcher, used to separate ALIKED features from LightGlue's learned matching)."""
    d = torch.cdist(da, db)
    v01, i01 = d.topk(2, dim=1, largest=False)
    i10 = d.argmin(dim=0)
    ok = (v01[:, 0] < ratio * v01[:, 1]) & (i10[i01[:, 0]] == torch.arange(len(da), device=da.device))
    idx = torch.nonzero(ok)[:, 0]
    return torch.stack([idx, i01[idx, 0]], 1)


def aliked_lightglue_into_db(db_path, image_dir, names, work_dir, matcher="exhaustive", overlap=10,
                             max_kpts=4096, matching="lightglue"):
    ext, lg = _models(max_kpts)
    db = pycolmap.Database.open(str(db_path))
    name2id = {im.name: im.image_id for im in db.read_all_images()}
    db.clear_descriptors()
    db.clear_keypoints()
    feats = {}
    for n in names:
        x = torch.from_numpy(np.asarray(Image.open(Path(image_dir) / n).convert("RGB"))).permute(2, 0, 1)
        x = x.float().div(255)[None].to(_dev())
        f = ext(x)[0]
        kp = f.keypoints.detach().float()
        feats[n] = (kp, f.descriptors.detach().float(), torch.tensor([[x.shape[-1], x.shape[-2]]], device=_dev()))
        # COLMAP keypoint convention: pixel centre at +0.5
        db.write_keypoints(name2id[n], (kp.cpu().numpy() + 0.5).astype(np.float32))
    if matcher == "exhaustive":
        pairs = list(itertools.combinations(names, 2))
    else:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, min(i + 1 + overlap, len(names)))]
    for a, b in pairs:
        ka, da, sa = feats[a]; kb, dbb, sb = feats[b]
        if matching == "lightglue":
            out = lg({"image0": {"keypoints": ka[None], "descriptors": da[None], "image_size": sa},
                      "image1": {"keypoints": kb[None], "descriptors": dbb[None], "image_size": sb}})
            m = out["matches"][0].cpu().numpy().astype(np.uint32)
        else:
            m = _mnn_ratio(da.detach(), dbb.detach()).detach().cpu().numpy().astype(np.uint32)
        if len(m):
            db.write_matches(name2id[a], name2id[b], m)
    db.close()
    pairs_txt = Path(work_dir) / "pairs.txt"
    pairs_txt.write_text("\n".join(f"{a} {b}" for a, b in pairs))
    pycolmap.verify_matches(db_path, pairs_txt)
