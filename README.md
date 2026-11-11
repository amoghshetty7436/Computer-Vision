# Where does Structure-from-Motion break? A stress test of COLMAP

CS 5672 group project (Cornell, Fall 2026). **Paper under test:** COLMAP, *Structure-from-Motion Revisited*
(Schönberger & Frahm, CVPR 2016), run through COLMAP 4 / `pycolmap`. **Comparisons:** global SfM (GLOMAP,
built into COLMAP 4) and learned features (ALIKED + LightGlue).

COLMAP was evaluated on clean internet photo collections. We test it in settings the paper does not cover:
real bad weather, desert conditions, image corruptions, generative AI edits, super-resolved inputs, and
crowds of tourists. For each setting we state in advance why it might fail and why it might still work.

* Results: [`results/summary.md`](results/summary.md) (auto-generated) and figures in [`results/figures/`](results/figures)
* Related work and hypotheses from the literature: [`docs/related_work.md`](docs/related_work.md)
* Example perturbations: [`docs/perturbation_examples.jpg`](docs/perturbation_examples.jpg), Boreas conditions: [`docs/boreas_contact_sheet.jpg`](docs/boreas_contact_sheet.jpg)

## One pipeline, one set of metrics

Every run uses the same pipeline, `sfmrob/sfm.py`: SIFT (8192 features), exhaustive matching, COLMAP
incremental mapping, a single shared camera, and EXIF focal prior (or known intrinsics for Boreas). The only
thing that changes between runs is the input images, or the one option being studied (E6).

| Metric | Meaning |
|---|---|
| **pose AUC@5/10/20°** (main) | IMC-style: for *all* input image pairs, `max(rotation error, translation-direction error)` of the relative pose vs ground truth. A pair with an unregistered image counts as a failure, so the AUC penalises both inaccuracy and failure to register. |
| registration rate | fraction of input images in the largest model (reported separately, following Bonilla et al. 2024) |
| ATE (rel.) | camera-centre RMSE after robust Sim(3) alignment, divided by scene extent |
| #points, track length, reprojection error | structure quality |

**Ground truth.**
* COLMAP scenes: the authors' full-resolution reconstruction from *all* images.
* Boreas: post-processed GNSS/INS poses. We validated the Euler-angle convention against a clean SfM run:
  only one of 12 conventions agrees, with a median relative-rotation error of 0.09° (`scripts/check_boreas_convention.py`).

**Rigour.**
* Each condition runs with 3 seeds, which change both the corruption RNG and COLMAP's RANSAC seed.
* Clean-run variance is reported. On Gerrard Hall, clean AUC@10 is 0.969 / 0.972 / 0.965 across seeds.
* Tables report 95% bootstrap CIs.
* Corruptions are applied *after* resizing to the working resolution, so a severity level means the same thing on every scene.

## Datasets

| Data | Used for | Why |
|---|---|---|
| COLMAP *Gerrard Hall*, *South Building* (+ *Person Hall* on the GPU cluster) | E1s–E6 | standard scenes with a reference reconstruction |
| **Boreas** (U. Toronto): the same 8 km route driven ~45 times over a year | E1-real, E7 | *real* sun / overcast / snow-covered / snowfall / rain / night at the **same places**, with survey-grade poses in one shared ENU frame |
| COCO person masks (coco128-seg) | E4 | real people cut-outs for crowds |

Boreas conditions were labelled by looking at a frame of every run (contact sheet in `docs/`). We evaluate 4 fixed
places along the route (0.5, 1.5, 2.6 and 3.5 km): 40 frames spaced 1.5 m apart, car hood cropped.

## Experiments, with why it might fail / why it might work

### E1  Weather- and desert-stratified evaluation *(TA #1)*
*Real:* SfM per Boreas condition at the same four places. *Synthetic:* depth-aware fog (Koschmieder model,
depth from the reference sparse points), rain (streaks + veiling + lens drops), snow, night (linear-light
exposure drop + photon/read noise + ISP gain), and three desert conditions: **sandstorm** (wavelength-dependent
scattering + orange airlight + wind-blown grains), **heat shimmer** (turbulent refraction field), and **harsh sun**
(clipping + glare). *Gap analysis:* the same synthetic weather applied to the sunny Boreas runs, compared with the real strata.
* **May fail:** scattering kills contrast, so SIFT's DoG peak threshold drops keypoints. Rain and snow add moving
  "features". Night photon noise breaks descriptors. Heat shimmer bends geometry *differently in every view*,
  which violates the static-scene assumption at the pixel level.
* **May work:** fog is mostly a monotone contrast change near the camera. RANSAC rejects moving particles.
  Sattler et al. (2018) found daytime rain and snow barely matter. Shimmer is sub-pixel at mild levels.

### E2  Escalating noise: analytic vs generative *(TA #2)*
*Analytic,* 5 levels each: Gaussian, shot (Poisson), salt & pepper, Gaussian/defocus/motion blur, JPEG,
pixelation, under/over-exposure, contrast.
*Generative:*
* Real-ESRGAN "regeneration": downsample, then let the GAN re-synthesise the detail.
* LaMa inpainting of free-form holes covering 5–45% of the image, with a **blackout control** using the same masks
  filled with grey. This separates *information removed* from *plausible-but-wrong information added*.
* SD-Turbo img2img at strength 0.1–0.6 (GPU).
* **May fail:** noise creates spurious DoG extrema. Blur removes the small scales. Generative models invent
  texture that looks matchable but is inconsistent across views, which can produce *confident wrong* matches.
* **May work:** SIFT's scale space and ratio test discard ambiguous matches. Inpainting changes only part of the image.

### E3  Super-resolution before SfM *(TA #3)*
Downsample ×2/×4/×8, then run SfM on (a) the low-res images directly, (b) bicubic, (c) bicubic + unsharp,
(d) Real-ESRGAN, (e) SD ×4 upscaler (GPU), and compare with native resolution.
* **May fail:** SR hallucinates high-frequency detail independently per view, so new keypoints are not
  repeatable and can create outlier tracks. Bicubic adds no information.
* **May work:** more pixels mean more octaves and keypoints. GAN SR sharpens real edges (Pashaei 2020 reports ~6x
  more keypoints). Prior work never measured *pose accuracy*, which we do.

### E4  Tourist crowds and the downstream effect *(TA #4)*
Real COCO people (2 to 40 per image) pasted with perspective-consistent size and colour matching, and
**re-sampled per view** (tourists move). Mitigations: (i) COLMAP feature masks on people, (ii) LaMa removal of people.
* **May fail:** people are textured, so they attract keypoints and occlude the facade. At high density whole
  views become mostly people, and removal leaves hallucinated patches.
* **May work:** moving people are inconsistent across views, so RANSAC rejects them, and the static background
  still dominates at moderate density.

### E5  (extra) How many bad photos does it take?
Corrupt only 10/25/50/75/100% of the images (motion blur, LaMa edits, crowds, night).
Crowd-sourced collections are partly bad, not all bad. Do bad images just fail to register, or do they poison the model?

### E6  (extra) Is the failure about SIFT, or about incremental SfM?
Hard conditions × {SIFT, ALIKED+LightGlue} × {incremental COLMAP, global GLOMAP}. No prior study crosses
extractor × back-end under controlled degradations (see `docs/related_work.md`).

### E7  (extra) Cross-season localisation
Build a map from a sunny Boreas run, then add 20 frames from each other condition at the same place. Measure the
fraction of query images registered *into the sunny model* and the map↔query relative-pose AUC.
This is the AR / robot re-localisation use case.

## Running it

```bash
pip install -r requirements.txt
export SFMROB_DATA=~/sfmrob_data SFMROB_WORK=/tmp/sfmrob_work
python -m sfmrob.run --exp e2_analytic --scene gerrard-hall --seeds 0 1 2      # any of e1_synth_weather, e2_analytic, e2_generative, e3_superres, e4_crowds, e5_contamination, e6_methods
python scripts/boreas_experiments.py --mode weather   # or xseason, synth
python scripts/analyze.py                             # tables + figures -> results/
```
* `--full` uses all 5 severities and adds the diffusion conditions (GPU + Hugging Face).
* GPU-only parts on Colab: `colab/sfmrob_gpu_experiments.ipynb`.
* NERSC Perlmutter: `perlmutter/submit_all.sh` submits a dependency chain of Slurm arrays (≤ 5.7 A100-h, ≈ 3.8 CPU node-h).

## Layout
```
sfmrob/sfm.py          COLMAP wrapper (one pipeline for all runs)
sfmrob/metrics.py      pose AUC, ATE, structure metrics
sfmrob/data.py         COLMAP scenes, Boreas (S3), reference poses, depth proxies
sfmrob/perturb/        analytic.py  weather.py  generative.py (Real-ESRGAN, LaMa)  diffusion.py  crowds.py
sfmrob/learned.py      ALIKED + LightGlue into a COLMAP database
sfmrob/run.py          controlled-experiment runner (resumable, shardable)
scripts/               Boreas experiments, convention check, analysis
perlmutter/  colab/    cluster / notebook entry points
results/raw/           one CSV row per run
```
