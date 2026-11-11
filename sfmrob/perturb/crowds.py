"""Tourist crowds: paste *real* people (COCO polygon cut-outs) into each view.

People are re-sampled independently per image, i.e. they move between shots exactly
like tourists do in internet photo collections. Placement is ground-anchored with a
simple perspective prior (people lower in the frame are larger), and soft-edged
alpha compositing + colour matching keeps them from being trivially separable.

Returns both the composited image and the binary mask of inserted people, which is
used by the mitigation conditions (COLMAP feature masks, LaMa removal).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..data import fetch_coco128


@lru_cache(1)
def person_bank(min_h=120):
    root = fetch_coco128()
    bank = []
    for lab in sorted((root / "labels" / "train2017").glob("*.txt")):
        img_p = root / "images" / "train2017" / (lab.stem + ".jpg")
        im = None
        for line in lab.read_text().splitlines():
            v = line.split()
            if not v or v[0] != "0" or len(v) < 7:
                continue
            if im is None:
                if not img_p.exists():
                    break
                im = np.array(Image.open(img_p).convert("RGB"))
            H, W = im.shape[:2]
            poly = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
            x, y, w, h = cv2.boundingRect(poly)
            # keep reasonably large, mostly un-truncated, upright people
            if h < min_h or h < 1.3 * w or y + h > H - 2 or y < 1:
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            bank.append((im[y:y + h, x:x + w].copy(), m[y:y + h, x:x + w].copy()))
    return bank


def _paste(dst, mask_acc, rgb, alpha, cx, foot_y, height):
    h0, w0 = rgb.shape[:2]
    s = height / h0
    w, h = max(int(w0 * s), 2), max(int(h0 * s), 2)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a = cv2.GaussianBlur(a, (0, 0), max(0.6, h / 300))
    H, W = dst.shape[:2]
    x0, y0 = int(cx - w / 2), int(foot_y - h)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    sub = dst[ya:yb, xa:xb].astype(np.float32)
    p = rgb[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    al = a[ya - y0:yb - y0, xa - x0:xb - x0, None]
    # colour/brightness matching to local background (mean/std transfer, half strength)
    if al.sum() > 10:
        mu_p = (p * al).sum((0, 1)) / al.sum()
        mu_b = sub.mean((0, 1))
        p = p + 0.5 * (mu_b - mu_p)
    dst[ya:yb, xa:xb] = np.clip(sub * (1 - al) + p * al, 0, 255).astype(np.uint8)
    mask_acc[ya:yb, xa:xb] |= (al[..., 0] > 0.3)


def add_crowd(img, n_people, rng, horizon=0.45, min_frac=0.12, max_frac=0.55):
    bank = person_bank()
    out = img.copy()
    H, W = img.shape[:2]
    mask = np.zeros((H, W), bool)
    # paste far people first so near people occlude them
    foots = np.sort(rng.uniform(horizon * H + 0.1 * H, H * 1.05, n_people))
    for fy in foots:
        t = (fy - horizon * H) / ((1.05 - horizon) * H)
        height = (min_frac + (max_frac - min_frac) * t) * H * rng.uniform(0.85, 1.15)
        rgb, a = bank[int(rng.integers(len(bank)))]
        if rng.random() < 0.5:
            rgb, a = rgb[:, ::-1], a[:, ::-1]
        _paste(out, mask, np.ascontiguousarray(rgb), np.ascontiguousarray(a), rng.uniform(0, W), fy, height)
    return out, mask


CROWD_LEVELS = {1: 2, 2: 5, 3: 10, 4: 20, 5: 40}


def crowd(img, s, rng, depth=None):
    return add_crowd(img, CROWD_LEVELS[s], rng)[0]
