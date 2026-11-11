"""Dataset preparation: download, subsample, resize, and load reference poses.

Scenes
------
* COLMAP benchmark scenes (gerrard-hall, south-building, person-hall): released on
  GitHub with a full-resolution COLMAP reconstruction from *all* images, which we use
  as pseudo ground truth for pose error.
* Boreas (https://www.boreas.utias.utoronto.ca, public S3 bucket): the same driving
  route in Toronto repeated over a year in sun, overcast, rain, snow and night, with
  post-processed GNSS/INS camera poses. Used for real weather-stratified evaluation.
* coco128-seg (Ultralytics GitHub release): 128 COCO images with polygon masks,
  used as a pool of real person cut-outs for the tourist-crowd experiment.
"""
from __future__ import annotations

import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(os.environ.get("SFMROB_DATA", Path.home() / "sfmrob_data"))

COLMAP_SCENES = {
    "gerrard-hall": "https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip",
    "south-building": "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip",
    "person-hall": "https://github.com/colmap/colmap/releases/download/3.11.1/person-hall.zip",
}
COCO128_SEG = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"
BOREAS = "https://boreas.s3.amazonaws.com/"


def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def fetch_colmap_scene(name: str) -> Path:
    root = DATA_ROOT / name
    if not (root / "images").exists():
        z = _download(COLMAP_SCENES[name], DATA_ROOT / f"{name}.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_ROOT)
        z.unlink()
    return root


def fetch_coco128() -> Path:
    root = DATA_ROOT / "coco128-seg"
    if not root.exists():
        z = _download(COCO128_SEG, DATA_ROOT / "coco128-seg.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_ROOT)
        z.unlink()
    return root


def load_colmap_reference(scene_root: Path) -> dict:
    """Reference poses name -> (R_cw, t_cw) from the released reconstruction."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(scene_root / "sparse"))
    from .metrics import poses_from_reconstruction
    return poses_from_reconstruction(rec)


def resize_to(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    s = max_side / max(w, h)
    if s >= 1:
        return im
    return im.resize((round(w * s), round(h * s)), Image.LANCZOS)


def prepare_colmap_subset(name: str, n_images: int, max_side: int, out: Path) -> list[str]:
    """Evenly subsample n_images (capture order) and resize. Returns image names."""
    root = fetch_colmap_scene(name)
    ref = load_colmap_reference(root)
    names = sorted(n for n in os.listdir(root / "images") if n in ref)
    idx = np.linspace(0, len(names) - 1, n_images).round().astype(int)
    sel = [names[i] for i in sorted(set(idx))]
    out.mkdir(parents=True, exist_ok=True)
    for n in sel:
        p = out / n
        if not p.exists():
            src = Image.open(root / "images" / n)
            exif = src.info.get("exif", b"")  # keep EXIF so COLMAP gets the same focal prior
            resize_to(src.convert("RGB"), max_side).save(p, quality=95, exif=exif)
    return sel


# ----------------------------------------------------------------------------- Boreas
def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def boreas_camera_poses(seq: str) -> np.ndarray:
    """Columns: GPSTime(us), easting, northing, altitude, ..., roll, pitch, heading."""
    cache = DATA_ROOT / "boreas" / seq / "camera_poses.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(f"{BOREAS}{seq}/applanix/camera_poses.csv"))
    return np.loadtxt(cache, delimiter=",", skiprows=1)


def boreas_intrinsics(seq: str) -> dict:
    txt = _get(f"{BOREAS}{seq}/calib/camera0_intrinsics.yaml").decode()
    f = [float(x) for x in re.search(r"focal_length: \[([^\]]*)\]", txt).group(1).split(",")]
    c = [float(x) for x in re.search(r"principal_points: \[([^\]]*)\]", txt).group(1).split(",")]
    return dict(fx=f[0], fy=f[1], cx=c[0], cy=c[1])


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def boreas_R_world_cam(roll, pitch, heading):
    """World(ENU)-from-camera rotation. Convention selected empirically by
    scripts/check_boreas_convention.py: of the 12 candidate Euler orders/transposes, only
    (Rz(heading) Ry(pitch) Rx(roll))^T agrees with a clean COLMAP model
    (median relative-rotation error 0.09 deg vs >= 3.9 deg for all others)."""
    return (_rot("z", heading) @ _rot("y", pitch) @ _rot("x", roll)).T


def boreas_select(seq: str, anchor_xy, anchor_heading: float, n_frames: int = 40,
                  spacing_m: float = 1.5, before_m: float = 30.0, max_dist: float = 15.0):
    """Indices of n_frames spaced spacing_m apart, centred on the pass of `seq` that goes
    through anchor_xy (ENU, shared by all Boreas sequences) with a similar heading."""
    P = boreas_camera_poses(seq)
    xy = P[:, 1:3]
    motion_heading = np.arctan2(P[:, 5], P[:, 4])  # direction of travel from ENU velocity
    dh = np.abs(np.angle(np.exp(1j * (motion_heading - anchor_heading))))
    dist = np.linalg.norm(xy - np.asarray(anchor_xy)[None], axis=1)
    cand = np.where((dh < np.radians(45)) & (dist < max_dist))[0]
    if len(cand) == 0:
        return None
    i0 = cand[np.argmin(dist[cand])]
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, 1:4], axis=0), axis=1))])
    targets = d[i0] - before_m + spacing_m * np.arange(n_frames)
    idx = np.unique(np.clip(np.searchsorted(d, targets), 0, len(P) - 1))
    return idx


def boreas_fetch(seq: str, idx, max_side: int, out: Path, crop_bottom: float = 0.18, workers: int = 8):
    """Download frames idx (crop car hood, resize). Returns (names, ref_poses, intrinsics).
    Ref poses are in the ENU frame shared by all sequences."""
    import concurrent.futures as cf
    P = boreas_camera_poses(seq)
    K = boreas_intrinsics(seq)
    W, H = 2448, 2048
    Hc = int(round(H * (1 - crop_bottom)))
    s = max_side / max(W, Hc)
    out.mkdir(parents=True, exist_ok=True)

    def fetch(i):
        ts = int(P[i, 0])
        name = f"{seq[7:]}_{ts}.jpg"
        p = out / name
        if not p.exists():
            im = Image.open(io.BytesIO(_get(f"{BOREAS}{seq}/camera/{ts}.png"))).convert("RGB")
            im = im.crop((0, 0, W, Hc)).resize((round(W * s), round(Hc * s)), Image.LANCZOS)
            tmp = p.with_suffix(f".{os.getpid()}.tmp.jpg")
            im.save(tmp, quality=95)
            os.replace(tmp, p)  # atomic: concurrent Slurm tasks may fetch the same frame
        return i, name

    names, ref = [], {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, name in ex.map(fetch, idx):
            R_wc = boreas_R_world_cam(P[i, 7], P[i, 8], P[i, 9])
            C = P[i, 1:4]
            ref[name] = (R_wc.T, -R_wc.T @ C)
            names.append(name)
    intr = dict(f=0.5 * (K["fx"] + K["fy"]) * s, cx=K["cx"] * s, cy=K["cy"] * s,
                width=round(W * s), height=round(Hc * s))
    return names, ref, intr

def colmap_depth_maps(name: str, names: list[str], image_dir: Path) -> dict:
    """Per-image depth proxies from the *reference* sparse model (projected 3D points,
    densified). Used only to make scattering effects depth-dependent."""
    import pycolmap
    from .perturb.weather import depth_from_points
    root = fetch_colmap_scene(name)
    rec = pycolmap.Reconstruction(str(root / "sparse"))
    out = {}
    for im in rec.images.values():
        if im.name not in names:
            continue
        w, h = Image.open(image_dir / im.name).size
        cam = rec.cameras[im.camera_id]
        s = w / cam.width
        uv, z = [], []
        T = im.cam_from_world().matrix()
        for p2 in im.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                z.append(float((T[:3, :3] @ X + T[:3, 3])[2]))
                uv.append(np.asarray(p2.xy) * s)
        out[im.name] = depth_from_points(h, w, np.array(uv).reshape(-1, 2), np.array(z))
    return out
