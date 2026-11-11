"""Diffusion-based generative perturbations (GPU tier; needs Hugging Face access).

* sd_img2img: SD-Turbo image-to-image at increasing strength. Low strength keeps the
  layout and re-synthesises texture; high strength starts to change geometry.
  This is "generative noise": each view is re-imagined independently, so fine detail
  is plausible per view but inconsistent across views.
* sd_x4_upscale: Stable Diffusion x4 upscaler (latent diffusion SR), the generative
  counterpart to Real-ESRGAN in the super-resolution experiment.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

IMG2IMG_MODEL = "stabilityai/sd-turbo"
UPSCALER_MODEL = "stabilityai/stable-diffusion-x4-upscaler"
STRENGTH = [0.1, 0.2, 0.3, 0.45, 0.6]


@lru_cache(1)
def _img2img():
    from diffusers import AutoPipelineForImage2Image
    # fp32: the fp16 VAE overflows at ~1024 px and returns black/NaN images
    p = AutoPipelineForImage2Image.from_pretrained(IMG2IMG_MODEL, torch_dtype=torch.float32)
    p.set_progress_bar_config(disable=True)
    return p.to("cuda")


@lru_cache(1)
def _upscaler():
    from diffusers import StableDiffusionUpscalePipeline
    p = StableDiffusionUpscalePipeline.from_pretrained(UPSCALER_MODEL, torch_dtype=torch.float32)
    p.set_progress_bar_config(disable=True)
    p.enable_attention_slicing()
    return p.to("cuda")


def _to_mult(img, m=64):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(m, w // m * m), max(m, h // m * m)), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def sdturbo_img2img(img, s, rng, depth=None, steps=10):
    h, w = img.shape[:2]
    x = Image.fromarray(_to_mult(img))
    g = torch.Generator("cuda").manual_seed(int(rng.integers(2**31)))
    out = _img2img()(prompt="a photo", image=x, strength=STRENGTH[s - 1], num_inference_steps=steps,
                     guidance_scale=0.0, generator=g).images[0]
    out = np.asarray(out)
    if out.mean() < 3:  # guard against silent NaN/black outputs
        raise RuntimeError("diffusion img2img returned a black image")
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)


@torch.no_grad()
def sd_x4_upscale(low, factor, tile=192):
    """Tiled SD x4 upscaling (tiles keep memory bounded); x8 = two passes is too slow,
    so x8 is done as SD x4 followed by bicubic x2."""
    h, w = low.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.uint8)
    pipe = _upscaler()
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            patch = low[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            up = pipe(prompt="a high quality photo", image=Image.fromarray(patch), num_inference_steps=20,
                      noise_level=20).images[0]
            out[y0 * 4:y0 * 4 + ph * 4, x0 * 4:x0 * 4 + pw * 4] = np.asarray(up)[:ph * 4, :pw * 4]
    if factor == 4:
        return out
    if factor == 2:
        return cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(out, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
