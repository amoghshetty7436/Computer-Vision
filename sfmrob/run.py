"""Experiment runner for the controlled (perturbation) experiments E1s-E6.

Every run = (scene, condition, seed, SfM config). For each run we
  1. take the clean resized subset of the scene,
  2. corrupt a fraction `frac` of the images with `perturb` at `severity`
     (deterministic per image/seed),
  3. optionally apply a mitigation (masks / LaMa removal),
  4. run the *same* COLMAP pipeline, and
  5. evaluate against the reference reconstruction.
Rows are appended to a CSV; finished rows are skipped, so jobs are resumable and can be
sharded across Slurm array tasks with --shard i/N.

Example:
  python -m sfmrob.run --exp e2_analytic --scene gerrard-hall --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import data as D
from .metrics import evaluate
from .sfm import SfMConfig, run_sfm
from .perturb.analytic import ANALYTIC
from .perturb.weather import WEATHER
from .perturb import generative as G
from .perturb import crowds as C

SEV_ALL = [1, 2, 3, 4, 5]


def _registry():
    reg = {}
    reg.update(ANALYTIC)
    reg.update(WEATHER)
    reg.update(G.GENERATIVE)
    reg["crowd"] = C.crowd
    return reg


# ----------------------------------------------------------------------------- experiment grids
def conditions(exp: str, full: bool) -> list[dict]:
    sev = SEV_ALL if full else [1, 3, 5]
    cfg = dict(features="sift", mapper="incremental")
    out = [dict(perturb="clean", severity=0, frac=1.0, mitigation="none", **cfg)]
    if exp == "e1_synth_weather":
        for p in ["fog", "rain", "snow", "night", "sandstorm", "heat_shimmer", "harsh_sun"]:
            out += [dict(perturb=p, severity=s, frac=1.0, mitigation="none", **cfg) for s in sev]
    elif exp == "e2_analytic":
        ps = list(ANALYTIC) if full else ["gaussian_noise", "shot_noise", "gaussian_blur", "motion_blur",
                                          "jpeg", "pixelate", "underexpose", "contrast"]
        for p in ps:
            out += [dict(perturb=p, severity=s, frac=1.0, mitigation="none", **cfg) for s in sev]
    elif exp == "e2_generative":
        for p in ["regen_esrgan", "inpaint_lama", "blackout"]:
            out += [dict(perturb=p, severity=s, frac=1.0, mitigation="none", **cfg) for s in sev]
        if full:  # diffusion img2img (GPU + HF): strength ladder
            out += [dict(perturb="sd_img2img", severity=s, frac=1.0, mitigation="none", **cfg) for s in sev]
    elif exp == "e3_superres":
        for f in ([2, 4, 8] if full else [4, 8]):
            for m in ["lowres", "bicubic", "unsharp_bicubic", "realesrgan"] + (["sd_x4"] if full and f >= 4 else []):
                out.append(dict(perturb=f"sr_{m}", severity=f, frac=1.0, mitigation="none", **cfg))
    elif exp == "e4_crowds":
        for s in sev if not full else SEV_ALL:
            for mit in ["none", "mask", "lama_remove"]:
                out.append(dict(perturb="crowd", severity=s, frac=1.0, mitigation=mit, **cfg))
    elif exp == "e5_contamination":
        for p, s in [("motion_blur", 4), ("inpaint_lama", 4), ("crowd", 5), ("night", 4)]:
            for fr in [0.1, 0.25, 0.5, 0.75]:
                out.append(dict(perturb=p, severity=s, frac=fr, mitigation="none", **cfg))
            out.append(dict(perturb=p, severity=s, frac=1.0, mitigation="none", **cfg))
    elif exp == "e6_methods":
        hard = [("clean", 0), ("motion_blur", 3), ("gaussian_noise", 4), ("night", 3), ("fog", 5),
                ("inpaint_lama", 3), ("crowd", 4), ("sr_realesrgan", 8)]
        out = []
        for feat in ["sift", "aliked_lg"]:
            for mapper in ["incremental", "global"]:
                for p, s in hard:
                    out.append(dict(perturb=p, severity=s, frac=1.0, mitigation="none",
                                    features=feat, mapper=mapper))
    else:
        raise ValueError(exp)
    return out


def cond_id(c: dict) -> str:
    return f"{c['perturb']}_s{c['severity']}_f{c['frac']}_{c['mitigation']}_{c['features']}-{c['mapper']}"


def _img_rng(name: str, c: dict, seed: int):
    h = hashlib.md5(f"{name}|{c['perturb']}|{c['severity']}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


def make_condition_images(clean_dir: Path, names: list[str], c: dict, seed: int, out_dir: Path,
                          depths: dict | None) -> Path | None:
    """Write the perturbed image set. Returns a mask dir if a masking mitigation applies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = None
    reg = _registry()
    # which images are corrupted (deterministic subset per seed)
    rng_sel = np.random.default_rng(seed + 12345)
    k = int(round(c["frac"] * len(names)))
    corrupted = set(rng_sel.permutation(names)[:k].tolist())
    p = c["perturb"]
    if c["mitigation"] == "mask":
        mask_dir = out_dir.parent / (out_dir.name + "_masks")
        mask_dir.mkdir(exist_ok=True)
    for n in names:
        src = Image.open(clean_dir / n)
        exif = src.info.get("exif", b"")
        img = np.array(src.convert("RGB"))
        rng = _img_rng(n, c, seed)
        person_mask = None
        if p == "clean" or n not in corrupted:
            res = img
        elif p.startswith("sr_"):
            method, f = p[3:], c["severity"]
            h, w = img.shape[:2]
            low = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
            res = low if method == "lowres" else (
                _sd_upscale(low, f) if method == "sd_x4" else G.sr_upscale(low, f, method))
            if method != "lowres" and res.shape[:2] != img.shape[:2]:
                res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)
        elif p == "crowd":
            res, person_mask = C.add_crowd(img, C.CROWD_LEVELS[c["severity"]], rng)
        elif p == "sd_img2img":
            from .perturb.diffusion import sd_img2img
            res = sd_img2img(img, c["severity"], rng)
        else:
            res = reg[p](img, c["severity"], rng, depths.get(n) if depths else None) \
                if p in WEATHER else reg[p](img, c["severity"], rng)
        if person_mask is not None and c["mitigation"] == "lama_remove":
            dil = cv2.dilate(person_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            res = G.lama_inpaint(res, dil)
        if mask_dir is not None:
            m = np.full(img.shape[:2], 255, np.uint8)
            if person_mask is not None:
                m[cv2.dilate(person_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
            cv2.imwrite(str(mask_dir / f"{n}.png"), m)
        Image.fromarray(res).save(out_dir / n, quality=95, exif=exif)
    return mask_dir


def _sd_upscale(low, f):
    from .perturb.diffusion import sd_x4_upscale
    return sd_x4_upscale(low, f)


GPU_GEN = {"regen_esrgan", "inpaint_lama", "sd_img2img", "sr_realesrgan", "sr_sd_x4"}


def needs_gpu_gen(c):
    return c["perturb"] in GPU_GEN or c["mitigation"] == "lama_remove" or \
        False


def needs_gpu_sfm(c):
    return c["features"] == "aliked_lg"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--scene", default="gerrard-hall")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--full", action="store_true", help="full severity grid + GPU/diffusion conditions")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--max_features", type=int, default=8192)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--work", default=os.environ.get("SFMROB_WORK", "/tmp/sfmrob_work"))
    ap.add_argument("--keep", action="store_true", help="keep perturbed images and models")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="substring filter on condition id")
    ap.add_argument("--phase", default="all", choices=["all", "gen", "cpu", "gpu_sfm"],
                    help="gen: only synthesise images that need a GPU model (no SfM); "
                         "cpu: SfM for everything except learned-feature runs (uses cached GPU images); "
                         "gpu_sfm: learned-feature (ALIKED+LightGlue) runs")
    a = ap.parse_args(argv)

    work = Path(a.work)
    base = work / f"{a.scene}_{a.n}_{a.size}"
    clean_dir = base / "clean"
    names = D.prepare_colmap_subset(a.scene, a.n, a.size, clean_dir)
    ref = D.load_colmap_reference(D.fetch_colmap_scene(a.scene))
    conds = conditions(a.exp, a.full)
    if a.only:
        conds = [c for c in conds if a.only in cond_id(c)]
    if a.phase == "gen":
        conds = [c for c in conds if needs_gpu_gen(c) and not needs_gpu_sfm(c)]
    elif a.phase == "cpu":
        conds = [c for c in conds if not needs_gpu_sfm(c)]
    elif a.phase == "gpu_sfm":
        conds = [c for c in conds if needs_gpu_sfm(c)]
    depths = None
    if any(c["perturb"] in WEATHER for c in conds):
        depths = D.colmap_depth_maps(a.scene, names, clean_dir)

    i, N = map(int, a.shard.split("/"))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (f"{a.exp}__{a.scene}.csv" if N == 1 else f"{a.exp}__{a.scene}__{a.phase}{i:03d}.csv")
    done = set()
    for f in out_dir.glob(f"{a.exp}__{a.scene}*.csv"):
        prev = pd.read_csv(f)
        done |= set(zip(prev["cond"], prev["seed"]))
    # SD x4 upscaling is expensive and (nearly) deterministic: one image seed only
    jobs = [(c, s) for c in conds for s in a.seeds if not (c["perturb"] == "sr_sd_x4" and s != a.seeds[0])]
    jobs = [j for k, j in enumerate(jobs) if k % N == i]
    print(f"[{a.phase}] {len(jobs)} runs ({len([j for j in jobs if (cond_id(j[0]), j[1]) in done])} already done)", flush=True)
    for c, seed in jobs:
        cid = cond_id(c)
        if (cid, seed) in done:
            continue
        t0 = time.time()
        cache = base / "imgcache" / a.exp / f"{cid}_seed{seed}"
        run_dir = base / a.exp / f"{cid}_seed{seed}"
        mask_dir = None
        if c["perturb"] == "clean":
            img_dir = clean_dir
        elif needs_gpu_gen(c):
            img_dir = cache / "images"
            if not (cache / ".done").exists():
                if a.phase == "cpu":
                    print(f"  skip {cid} seed={seed}: GPU-generated images not ready", flush=True)
                    continue
                shutil.rmtree(cache, ignore_errors=True)
                mask_dir = make_condition_images(clean_dir, names, c, seed, img_dir, depths)
                (cache / ".done").write_text("ok")
            elif c["mitigation"] == "mask":
                mask_dir = cache / "images_masks"
            if a.phase == "gen":
                print(f"  generated {cid} seed={seed} ({time.time() - t0:.0f}s)", flush=True)
                continue
        else:
            img_dir = run_dir / "images"
            mask_dir = make_condition_images(clean_dir, names, c, seed, img_dir, depths)
        cfg = SfMConfig(features=c["features"], mapper=c["mapper"], max_num_features=a.max_features,
                        seed=seed, use_gpu=a.gpu, num_threads=a.threads)
        res = run_sfm(img_dir, run_dir / "sfm", cfg, image_names=names, mask_dir=mask_dir)
        m = evaluate(res, ref, names)
        row = dict(exp=a.exp, scene=a.scene, cond=cid, seed=seed, **c, **m, wall_s=round(time.time() - t0, 1))
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"[{a.exp}] {cid} seed={seed} reg={m['n_reg']}/{m['n_input']} auc@10={m.get('auc@10', float('nan')):.3f} "
              f"({row['wall_s']}s)", flush=True)
        if not a.keep:
            shutil.rmtree(run_dir, ignore_errors=True)
            if mask_dir is not None and not str(mask_dir).startswith(str(cache)):
                shutil.rmtree(mask_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
