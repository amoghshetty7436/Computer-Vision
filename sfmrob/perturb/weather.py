"""Physically-motivated synthetic weather, incl. desert conditions.

Depth matters for scattering (fog / haze / sandstorm), so these functions take an
optional depth map. `depth_proxy` builds one without any learned model:
  * from the reference sparse reconstruction (project reference 3D points into the
    image and densify), or
  * from a flat ground-plane + sky model (used for driving data).

Model: Koschmieder atmospheric scattering  I = J * t + A * (1 - t),  t = exp(-beta d).
"""
from __future__ import annotations

import cv2
import numpy as np


def _f(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- depth proxies
def depth_from_points(h, w, uv: np.ndarray, z: np.ndarray, sky_factor=3.0):
    """Densify sparse depths (pixel uv, depth z) to a full map; pixels far from any
    point at the top of the image are treated as sky (sky_factor * max depth)."""
    D = np.zeros((h, w), np.float32)
    if len(z) < 10:
        return ground_plane_depth(h, w)
    uv = np.round(uv).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (z > 0)
    uv, z = uv[ok], z[ok]
    small = 4
    hs, ws = h // small, w // small
    acc = np.zeros((hs, ws), np.float32); cnt = np.zeros((hs, ws), np.float32)
    np.add.at(acc, (np.minimum(uv[:, 1] // small, hs - 1), np.minimum(uv[:, 0] // small, ws - 1)), z)
    np.add.at(cnt, (np.minimum(uv[:, 1] // small, hs - 1), np.minimum(uv[:, 0] // small, ws - 1)), 1)
    known = cnt > 0
    dmap = np.where(known, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    # nearest-neighbour fill via distance transform labels
    dist, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    lut = np.zeros(labels.max() + 1, np.float32)
    lut[labels[ys, xs]] = dmap[ys, xs]
    filled = lut[labels]
    # sky: rows above the highest 2nd-percentile reconstructed point, far from points
    top = np.percentile(ys, 2)
    sky = (np.arange(hs)[:, None] < top) & (dist > 6)
    filled[sky] = z.max() * sky_factor
    filled = cv2.GaussianBlur(filled, (0, 0), 3)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)


def ground_plane_depth(h, w, horizon=0.45, cam_height=1.8, f=None, far=300.0):
    f = f or 0.6 * w
    y = np.arange(h, dtype=np.float32)[:, None] - horizon * h
    d = np.where(y > 0, cam_height * f / np.maximum(y, 1e-3), far)
    return np.repeat(np.minimum(d, far), w, axis=1).astype(np.float32)


def _norm_depth(depth):
    """Log-compressed depth in ~[0, 1.3]: 1 = 95th percentile scene depth."""
    d = np.log1p(np.maximum(depth, 0)) / (np.log1p(np.percentile(depth, 95)) + 1e-6)
    return cv2.GaussianBlur(np.clip(d, 0, 1.3).astype(np.float32), (0, 0), 5)


# ----------------------------------------------------------------------------- weather
def fog(img, s, rng, depth=None):
    beta = [0.4, 0.8, 1.4, 2.2, 3.2][s - 1]
    d = _norm_depth(depth if depth is not None else ground_plane_depth(*img.shape[:2]))
    t = np.exp(-beta * d)[..., None]
    A = 0.85 + 0.05 * rng.random()
    return _u8(_f(img) * t + A * (1 - t))


def sandstorm(img, s, rng, depth=None):
    """Desert dust storm: wavelength-dependent scattering (blue attenuated more), orange
    airlight, contrast loss, and fine sand particles."""
    beta = [0.5, 1.0, 1.6, 2.4, 3.4][s - 1]
    d = _norm_depth(depth if depth is not None else ground_plane_depth(*img.shape[:2]))
    beta_rgb = beta * np.array([0.8, 1.0, 1.35], np.float32)
    t = np.exp(-d[..., None] * beta_rgb)
    A = np.array([0.80, 0.62, 0.38], np.float32) * (0.95 + 0.1 * rng.random())
    x = _f(img) * t + A * (1 - t)
    # suspended sand grains: small bright-brown speckles, motion-blurred by wind
    h, w = img.shape[:2]
    n = int([0.0005, 0.001, 0.002, 0.003, 0.004][s - 1] * h * w)
    layer = np.zeros((h, w), np.float32)
    layer[rng.integers(0, h, n), rng.integers(0, w, n)] = rng.uniform(0.3, 0.8, n)
    L = 7 + 2 * s
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1; k /= L
    layer = cv2.filter2D(layer, -1, k)
    x = x * (1 - layer[..., None]) + layer[..., None] * A
    return _u8(x)


def heat_shimmer(img, s, rng, depth=None):
    """Heat haze over hot ground: smooth random refractive displacement field, stronger
    for distant pixels near the horizon. Different per image (it is turbulent)."""
    amp = [0.75, 1.5, 2.5, 4.0, 6.0][s - 1]
    h, w = img.shape[:2]
    d = _norm_depth(depth if depth is not None else ground_plane_depth(h, w))
    weight = np.clip(d, 0, 1.5) / 1.5
    def field(scale):
        n = rng.normal(0, 1, (max(h // scale, 2), max(w // scale, 2))).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.GaussianBlur(n, (0, 0), scale / 2)
    dx = (field(16) + 0.5 * field(6)) * amp * weight
    dy = (field(16) + 0.5 * field(6)) * amp * weight * 0.6
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def harsh_sun(img, s, rng, depth=None):
    """Desert midday: over-exposure with highlight clipping, strong veiling glare and
    washed-out shadows (low dynamic range of bright sand)."""
    gain = [1.5, 2.0, 2.8, 3.8, 5.0][s - 1]
    x = _f(img) ** 2.2 * gain
    glare = cv2.GaussianBlur(np.clip(x - 0.8, 0, None), (0, 0), 25) * (0.5 + 0.2 * s)
    x = np.clip(x + glare, 0, 1) ** (1 / 2.2)
    return _u8(x)


def rain(img, s, rng, depth=None):
    """Rain streaks + atmospheric veiling + slight defocus from wet air + lens drops."""
    h, w = img.shape[:2]
    n = int([0.0008, 0.0015, 0.003, 0.005, 0.008][s - 1] * h * w)
    L = [10, 15, 20, 28, 36][s - 1]
    ang = rng.uniform(-20, 20)
    layer = np.zeros((h, w), np.float32)
    layer[rng.integers(0, h, n), rng.integers(0, w, n)] = rng.uniform(0.4, 1.0, n)
    k = np.zeros((L, L), np.float32); k[:, L // 2] = 1
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1)
    k = cv2.warpAffine(k, M, (L, L)); k /= k.sum() + 1e-6
    layer = cv2.filter2D(layer, -1, k) * 3
    x = _f(img) * (1 - 0.06 * s)  # overcast darkening
    x = fog(_u8(x), max(1, s - 2), rng, depth) if s >= 3 else _u8(x)
    x = _f(x)
    x = x * (1 - 0.6 * layer[..., None]) + 0.8 * layer[..., None]
    out = _u8(x)
    # water drops on the lens: blurred discs with local magnification
    nd = [0, 3, 6, 12, 20][s - 1]
    for _ in range(nd):
        r = int(rng.uniform(0.01, 0.04) * w)
        cx, cy = int(rng.integers(r, w - r)), int(rng.integers(r, h - r))
        patch = out[cy - r:cy + r, cx - r:cx + r]
        blurred = cv2.GaussianBlur(cv2.flip(patch, 0), (0, 0), r / 3)
        m = np.zeros(patch.shape[:2], np.float32)
        cv2.circle(m, (r, r), r - 1, 1, -1)
        m = cv2.GaussianBlur(m, (0, 0), 2)[..., None]
        out[cy - r:cy + r, cx - r:cx + r] = (patch * (1 - m) + blurred * m).astype(np.uint8)
    return out


def snow(img, s, rng, depth=None):
    h, w = img.shape[:2]
    x = _f(img)
    x = x * (1 - 0.08 * s) + 0.08 * s * 0.85  # whitish veiling
    out = x.copy()
    n_layers = 3
    for li in range(n_layers):
        n = int([0.0006, 0.0012, 0.002, 0.0035, 0.005][s - 1] * h * w / (li + 1))
        size = (li + 1) * 1.2
        layer = np.zeros((h, w), np.float32)
        layer[rng.integers(0, h, n), rng.integers(0, w, n)] = 1
        layer = cv2.GaussianBlur(layer, (0, 0), size) * size * 6
        L = 3 + 3 * li
        k = np.zeros((L, L), np.float32); k[:, L // 2] = 1; k /= L
        layer = np.clip(cv2.filter2D(layer, -1, k), 0, 1)
        out = out * (1 - layer[..., None]) + layer[..., None] * 0.95
    return _u8(out)


def night(img, s, rng, depth=None):
    """Low light: exposure drop in linear space, blue shift, photon + read noise,
    and the sensor/ISP's gain boost (which amplifies noise)."""
    gain = [0.25, 0.12, 0.06, 0.03, 0.015][s - 1]
    lin = _f(img) ** 2.2 * gain * np.array([0.85, 0.95, 1.1], np.float32)
    photons = 2000 * gain
    lin = rng.poisson(np.clip(lin, 0, 1) * photons * 50) / (photons * 50)
    lin = lin + rng.normal(0, 0.0015, lin.shape)
    boost = 0.25 / max(gain, 1e-3) ** 0.5  # auto-gain partially compensates
    lin = np.clip(lin * boost * 4, 0, 1)
    return _u8(lin ** (1 / 2.2))


WEATHER = {
    "fog": fog, "rain": rain, "snow": snow, "night": night,
    "sandstorm": sandstorm, "heat_shimmer": heat_shimmer, "harsh_sun": harsh_sun,
}
