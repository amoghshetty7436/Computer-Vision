"""Generative perturbations: content produced by a learned generative model.

* Real-ESRGAN x4 (GAN super-resolution) - hallucinates high-frequency texture.
  Used for (a) "regeneration" noise: downsample by k then re-synthesise detail, and
  (b) the super-resolution experiment.
* LaMa (big-lama, Fourier-conv inpainting GAN) - replaces masked regions with
  plausible but *view-inconsistent* content. Used for (a) generative inpainting
  noise with growing masked area, and (b) removing tourists in the crowd experiment.
* Diffusion variants (SD img2img / SD x4 upscaler / SD inpainting) live in
  `diffusion.py` and need a GPU + Hugging Face access (run on Perlmutter).

Weights are downloaded from the authors' GitHub releases.
"""
from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = Path(os.environ.get("SFMROB_MODELS", Path(os.environ.get("SFMROB_DATA", Path.home() / "sfmrob_data")) / "models"))
URLS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "big-lama.pt": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
}


def _weights(name):
    p = MODEL_DIR / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], p)
    return p


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- Real-ESRGAN
class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


@lru_cache(1)
def _esrgan():
    net = RRDBNet()
    sd = torch.load(_weights("RealESRGAN_x4plus.pth"), map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd), strict=True)
    return net.eval().to(_device())


@torch.no_grad()
def realesrgan_x4(img_u8: np.ndarray, tile: int = 256) -> np.ndarray:
    net = _esrgan()
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255).unsqueeze(0).to(_device())
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * 4, w * 4, device=x.device)
    pad = 10
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, yb = max(y0 - pad, 0), min(y0 + tile + pad, h)
            xa, xb = max(x0 - pad, 0), min(x0 + tile + pad, w)
            o = net(x[:, :, ya:yb, xa:xb])
            oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
            th, tw = min(tile, h - y0) * 4, min(tile, w - x0) * 4
            out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = o[:, :, oy:oy + th, ox:ox + tw]
    return (out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def sr_upscale(img_u8: np.ndarray, factor: int, method: str) -> np.ndarray:
    """Upscale a low-res image by `factor` (2,4,8) with the given method."""
    h, w = img_u8.shape[:2]
    target = (w * factor, h * factor)
    if method == "bicubic":
        return cv2.resize(img_u8, target, interpolation=cv2.INTER_CUBIC)
    if method == "lanczos":
        return cv2.resize(img_u8, target, interpolation=cv2.INTER_LANCZOS4)
    if method == "realesrgan":
        x = img_u8
        f = 1
        while f < factor:
            x = realesrgan_x4(x)
            f *= 4
        return cv2.resize(x, target, interpolation=cv2.INTER_AREA) if x.shape[1] != target[0] else x
    if method == "unsharp_bicubic":   # classical "enhancement" baseline
        up = cv2.resize(img_u8, target, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), factor * 0.6)
        return cv2.addWeighted(up, 1.8, blur, -0.8, 0)
    raise ValueError(method)


def regen_noise(img, s, rng, depth=None):
    """Generative 'regeneration' noise: downsample by k and let Real-ESRGAN re-synthesise
    the lost detail. Higher severity = more of the image content is hallucinated."""
    k = [1.5, 2, 3, 4, 6][s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w / k), 8), max(int(h / k), 8)), interpolation=cv2.INTER_AREA)
    up = realesrgan_x4(small)
    return cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- LaMa
@lru_cache(1)
def _lama():
    m = torch.jit.load(str(_weights("big-lama.pt")), map_location="cpu")
    return m.eval().to(_device())


@torch.no_grad()
def lama_inpaint(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask: HxW bool/uint8, True = fill."""
    h, w = img_u8.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    im = np.pad(img_u8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad((mask > 0).astype(np.float32), ((0, ph), (0, pw)), mode="reflect")
    x = torch.from_numpy(im).permute(2, 0, 1).float().div(255).unsqueeze(0).to(_device())
    m = torch.from_numpy(mk)[None, None].to(_device())
    out = _lama()(x, m)
    out = (out[0].permute(1, 2, 0).cpu().numpy()[:h, :w])
    out = np.clip(out * 255 if out.max() <= 1.5 else out, 0, 255).astype(np.uint8)
    keep = (mask[..., None] == 0)
    return np.where(keep, img_u8, out)


def random_freeform_mask(h, w, area_frac, rng):
    """Brush-stroke masks like LaMa/DeepFill training masks, until area_frac is covered."""
    m = np.zeros((h, w), np.uint8)
    while m.mean() < area_frac:
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(int(rng.integers(3, 8))):
            ang = rng.uniform(0, 2 * np.pi)
            L = rng.uniform(0.05, 0.2) * max(h, w)
            nx, ny = int(np.clip(x + L * np.cos(ang), 0, w - 1)), int(np.clip(y + L * np.sin(ang), 0, h - 1))
            cv2.line(m, (x, y), (nx, ny), 1, int(rng.uniform(0.02, 0.06) * max(h, w)))
            x, y = nx, ny
    return m.astype(bool)


def inpaint_noise(img, s, rng, depth=None):
    """Generative inpainting noise: LaMa hallucinates content in free-form holes."""
    frac = [0.05, 0.10, 0.20, 0.30, 0.45][s - 1]
    m = random_freeform_mask(*img.shape[:2], frac, rng)
    return lama_inpaint(img, m)


def blackout_noise(img, s, rng, depth=None):
    """Control for inpaint_noise: same masks, filled with grey instead of generated content.
    Isolates 'information removed' from 'plausible-but-wrong information added'."""
    frac = [0.05, 0.10, 0.20, 0.30, 0.45][s - 1]
    m = random_freeform_mask(*img.shape[:2], frac, rng)
    out = img.copy(); out[m] = 127
    return out


GENERATIVE = {"regen_esrgan": regen_noise, "inpaint_lama": inpaint_noise, "blackout": blackout_noise}
