# Where does Structure-from-Motion break? Results and conjectures

*Numbers are from `results/summary.md` (auto-generated). Still pending, on GPU: SD-Turbo img2img noise, SD ×4 super-resolution, and ALIKED + LightGlue.*

**Setup in one paragraph.** We run COLMAP (Schönberger & Frahm, CVPR 2016) through COLMAP 4 / pycolmap. The pipeline is
SIFT with 8192 features, exhaustive matching and incremental mapping. We corrupt the input photos in controlled ways
and measure **pose AUC@10°** over *all* image pairs: an unregistered image counts as a failure. A score of 1 means every
relative pose is within 10°. Scenes:
* *Gerrard Hall* (50 images) and *South Building* (40 images), both at 1024 px, with ground truth from COLMAP's
  full-resolution model.
* *Boreas*: the same Toronto route driven in 15 different runs across 6 weather/light strata, with GNSS/INS ground
  truth, evaluated at 4 fixed places.

Every synthetic condition was run with 3 seeds on 2 scenes (n = 6); Boreas has 3 seeds per run and place. Brackets are
95% bootstrap CIs. The clean baseline is AUC@10 = **0.959** (42 runs, std 0.047).

---

## E1 Weather and desert (TA ask #1)

**Real weather (Boreas)**

| condition | AUC@10 | registered |
|---|---|---|
| rain | 0.87 [0.75, 0.98] | 0.96 |
| snow on the ground | 0.88 [0.77, 0.96] | 0.91 |
| sunny | 0.74 [0.62, 0.84] | 0.76 |
| snowfall | 0.65 [0.33, 0.98] | 0.67 |
| overcast | 0.54 [0.38, 0.70] | 0.56 |
| **night** | **0.02 [0.01, 0.04]** | 0.51 |

* **Night is the only condition that breaks SfM.** Even when half of the night frames register, their poses are wrong.
* **Daytime rain and snow do not hurt.** This agrees with Sattler et al. (2018).
* **Variance between places is larger than between weathers.** One stretch of road (a fast, straight segment with
  little parallax) fails often in any weather, which is why "sunny" scores below rain and snow.
* *Conjecture:* for driving data, scene geometry and motion matter more than weather, except at night, where photon
  noise, headlights and motion blur destroy the descriptors.

**Synthetic weather on Gerrard Hall / South Building (AUC@10 at severity 1 / 3 / 5)**

| condition | s1 | s3 | s5 |
|---|---|---|---|
| heat shimmer (desert) | 0.97 | 0.97 | 0.77 |
| sandstorm (desert) | 0.97 | 0.76 | 0.20 |
| harsh sun (desert) | 0.98 | 0.34 | 0.16 |
| fog | 0.97 | 0.72 | 0.21 |
| rain | 0.90 | 0.44 | 0.16 |
| night | 0.85 | 0.37 | 0.005 |
| snow | 0.55 | 0.22 | 0.00 |

* **Desert:** heat shimmer is almost harmless until it is extreme. It bends pixels differently in each view, but by
  less than SIFT's localisation noise. Harsh sun (clipped highlights and glare) is the worst desert effect, because
  saturated regions have no gradients, so no keypoints form there.
* **Snow falling in front of the camera is the worst weather.** The flakes create thousands of blob-like DoG
  extrema that compete with real keypoints for the 8192-feature budget.

**Synthetic vs real (the same synthetic effects applied to sunny Boreas runs)**
* Synthetic fog and snow at full strength are *harsher* than any real Boreas condition (0.17 and 0.04, against
  0.65 to 0.88 for real snow).
* Synthetic night (0.33) is *milder* than real night (0.02).
* Corruption benchmarks therefore mis-rank conditions. Real night is the hard case, and simulating it with an exposure
  drop plus noise underestimates it.

## E2 Escalating noise: analytic vs generative (TA ask #2)

**Analytic (AUC@10 at s1 / s3 / s5):**

| corruption | s1 | s3 | s5 |
|---|---|---|---|
| contrast | 0.95 | 0.35 | 0.00 |
| pixelate | 0.95 | 0.29 | 0.04 |
| Gaussian blur | 0.93 | 0.13 | 0.00 |
| underexpose | 0.91 | 0.68 | 0.10 |
| Gaussian noise | 0.83 | 0.28 | 0.00 |
| motion blur | 0.60 | 0.003 | 0.00 |
| JPEG | 0.57 | 0.19 | 0.00 |
| shot (Poisson) noise | 0.31 | 0.07 | 0.00 |

* **Two kinds of failure.**
  - *Removal* (blur, pixelation, low contrast) mainly lowers registration: images simply drop out.
  - *Additive* noise and JPEG blocking keep registration high but lower accuracy. Mild JPEG registers 99% of images
    yet scores only 0.57, because block artefacts create repeatable but wrong keypoints at 8×8 block corners.
* Motion blur is the most damaging single corruption. It is anisotropic, so it shifts keypoints along the blur
  direction differently in each view.

**Generative (AUC@10 at s1 / s3 / s5, n = 6 each):**

| corruption | s1 | s3 | s5 |
|---|---|---|---|
| blackout control (holes filled grey) | 0.97 | 0.48 | 0.22 |
| LaMa inpainting (same holes, filled plausibly) | 0.97 | 0.56 | 0.26 |
| Real-ESRGAN regeneration (down ×1.5 / ×3 / ×6, GAN re-synthesises the detail) | **0.35** | 0.21 | 0.02 |

* **Inpainting is *not* worse than deleting the region.** LaMa fills slightly beat grey holes. SIFT finds few
  keypoints in the smooth, low-frequency fills, so the invented content is mostly ignored, not mismatched.
* **GAN regeneration is the most harmful generative edit.** With only ×1.5 downsampling, ESRGAN regeneration drops
  AUC to 0.35. The analytic equivalent (pixelate s1) keeps 0.95. The GAN replaces *every* fine texture with its own
  hallucination, so keypoints lose repeatability across views.
* *Conjecture:* how much a generative edit harms SfM depends on how much **high-frequency** content it invents, not on
  how much of the image it changes. Diffusion img2img (SD-Turbo, pending on GPU) should fall between the two,
  depending on strength.

## E3 Super-resolution before SfM (TA ask #3)

| | ×4 down then up | ×8 down then up |
|---|---|---|
| low-res input as is | 0.02 | 0.00 |
| bicubic | 0.31 | 0.03 |
| bicubic + unsharp | 0.32 | 0.05 |
| Real-ESRGAN | 0.11 | 0.00 |

* Classical upscaling helps a lot over the raw low-res input (0.02 → 0.31). The extra pixels give SIFT more octaves to
  work with.
* The GAN (Real-ESRGAN) is *worse* than bicubic: it hallucinates high-frequency texture independently in each view, so
  the new keypoints don't repeat across views.
* *Conjecture:* SR helps SfM only if it is multi-view consistent. Single-image generative SR, the usual
  "enhancement" step, actively hurts.

## E4 Tourist crowds (TA ask #4)

| people per image | no mitigation | mask people | LaMa remove |
|---|---|---|---|
| 2 | 0.94 | 0.97 | 0.97 |
| 10 | 0.30 (reg 0.94) | 0.40 | 0.38 |
| 40 | **0.04 (reg 0.89)** | 0.32 | 0.32 |

* **Crowds cause silent failure.** Almost every image still registers, but the poses are wrong. *Conjecture:* people are richly
  textured, so they win the feature budget. The COCO pool of cut-outs is finite, so the same person can appear in
  several views and give consistent-looking but wrong matches.
* Masking and LaMa removal help about equally (0.04 → 0.32), so the cheap fix (a person mask) is enough. Neither
  recovers the background the people hid. At 40 people per image that is a large part of the facade.

## E5 How many bad photos does it take? (extra)

| fraction of photos corrupted | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| motion blur | 0.60 | 0.29 | 0.03 | 0.00 |
| night | 0.80 | 0.68 | 0.35 | 0.09 |
| crowds | 0.77 | 0.66 | 0.30 | 0.02 |

(clean: 0.92)

* **Bad photos poison good ones.** With 10% of photos motion-blurred, registration falls to 0.83, which is more than the
  10% we blurred, and accuracy falls by a third.
* *Conjecture:* in incremental SfM, a blurred image registered early with a poor pose triangulates bad points. Later
  good images then register against those points. Filtering photos by quality before SfM is worth more than making
  each step robust.

## E6 Is the failure in SIFT or in incremental SfM? (extra)

| condition | SIFT + incremental | SIFT + global (GLOMAP) |
|---|---|---|
| clean | 0.96 | 0.38 |
| fog s5 | 0.20 | 0.41 |
| Gaussian noise s4 | 0.09 | 0.27 |
| night s3 | 0.37 | 0.39 |
| crowds s4 | 0.11 | 0.01 |

* GLOMAP fails on *clean* Gerrard Hall. It places 12 consecutive photos of one side of the building rotated by about
  135°, a symmetric-facade ambiguity that global rotation averaging cannot resolve (`docs/notes_glomap_symmetry.md`).
* Under fog and noise, though, global SfM degrades *less* than incremental. It uses all pairs at once instead of
  chaining registrations, so one bad step can't cascade.
* ALIKED + LightGlue (learned features): pending, runs on GPU.

## E7 Cross-season re-localisation (extra)

A map built from a sunny Boreas run, with 20 query frames from other conditions added at the same place:

| query condition | registered into the sunny map | cross AUC@10 |
|---|---|---|
| sunny (another day) | 1.00 | 0.90 |
| overcast | 0.97 | 0.90 |
| snow on the ground | 0.97 | 0.88 |
| rain | 0.88 | 0.83 |
| snowfall | 0.87 | 0.76 |
| **night** | **0.00** | **0.00** |

* Seasonal appearance change, even snow cover, is fine for SIFT at the same viewpoint.
* Night is a hard wall: not one of 240 night queries localised into a daytime map.

---

## Limitations
* The two COLMAP scenes are subsampled (50 / 40 images, 1024 px), which makes them harder than the full sets. The
  relative comparisons are what matter.
* Boreas strata were labelled visually, and overcast/sunny partly mixes with how hard each place is.
* Synthetic corruptions are models. E1-gap shows they can mis-rank real conditions.
