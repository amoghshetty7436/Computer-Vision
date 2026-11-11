"""Analytic (closed-form) image corruptions with 5 severity levels.

All functions: f(img_uint8_rgb, severity:int in 1..5, rng) -> img_uint8_rgb.
Severity tables are expressed relative to a ~1024 px long side, which is the
working resolution of all experiments (images are resized *before* corruption).
"""
from __future__ import annotations

import cv2
import numpy as np


def _f(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def gaussian_noise(img, s, rng):
    sigma = [0.02, 0.04, 0.08, 0.12, 0.18][s - 1]
    x = _f(img)
    return _u8(x + rng.normal(0, sigma, x.shape))


def shot_noise(img, s, rng):
    photons = [120, 60, 25, 12, 5][s - 1]
    x = _f(img)
    return _u8(rng.poisson(x * photons) / photons)


def salt_pepper(img, s, rng):
    p = [0.01, 0.03, 0.06, 0.10, 0.17][s - 1]
    x = img.copy()
    m = rng.random(x.shape[:2])
    x[m < p / 2] = 0
    x[m > 1 - p / 2] = 255
    return x


def gaussian_blur(img, s, rng):
    sigma = [1.0, 2.0, 3.0, 5.0, 8.0][s - 1]
    return cv2.GaussianBlur(img, (0, 0), sigma)


def defocus_blur(img, s, rng):
    r = [2, 4, 6, 9, 13][s - 1]
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1, -1)
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def motion_blur(img, s, rng):
    L = [5, 10, 20, 30, 45][s - 1]
    ang = rng.uniform(0, 180)
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def jpeg(img, s, rng):
    q = [40, 20, 10, 5, 2][s - 1]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def pixelate(img, s, rng):
    f = [2, 3, 4, 6, 8][s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def underexpose(img, s, rng):
    gain = [0.5, 0.3, 0.18, 0.1, 0.05][s - 1]
    x = _f(img) ** 2.2 * gain  # linear-light exposure change
    # sensor read noise becomes visible in the dark, plus 8-bit quantisation
    x = x + rng.normal(0, 0.002, x.shape)
    return _u8(np.clip(x, 0, 1) ** (1 / 2.2))


def overexpose(img, s, rng):
    gain = [2, 3, 4.5, 7, 11][s - 1]
    return _u8((_f(img) ** 2.2 * gain) ** (1 / 2.2))


def contrast(img, s, rng):
    c = [0.6, 0.45, 0.3, 0.2, 0.12][s - 1]
    x = _f(img)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * c + m)


ANALYTIC = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "salt_pepper": salt_pepper,
    "gaussian_blur": gaussian_blur,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "jpeg": jpeg,
    "pixelate": pixelate,
    "underexpose": underexpose,
    "overexpose": overexpose,
    "contrast": contrast,
}
