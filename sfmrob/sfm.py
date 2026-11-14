"""Thin, reproducible wrapper around COLMAP (via pycolmap >= 4.0).

One function, `run_sfm`, is used by every experiment so that the only thing that
changes between runs is the input images (or the explicitly varied option).
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pycolmap


@dataclass
class SfMConfig:
    features: str = "sift"          # sift | aliked_lg (LightGlue) | aliked_nn (mutual NN + ratio)
    matcher: str = "exhaustive"     # exhaustive | sequential
    mapper: str = "incremental"     # incremental (COLMAP) | global (GLOMAP inside COLMAP 4)
    max_num_features: int = 8192
    seq_overlap: int = 10           # only for sequential matching
    single_camera: bool = True
    camera_model: str = "SIMPLE_RADIAL"
    camera_params: str = ""         # e.g. "f,cx,cy,k" to fix a known prior
    num_threads: int = -1
    seed: int = 0
    use_gpu: bool = False

    def tag(self) -> str:
        return f"{self.features}-{self.matcher}-{self.mapper}"


@dataclass
class SfMResult:
    ok: bool
    n_input: int
    models: list = field(default_factory=list)   # list of pycolmap.Reconstruction (largest first)
    timings: dict = field(default_factory=dict)
    error: str = ""

    @property
    def best(self):
        return self.models[0] if self.models else None


def _extraction_options(cfg: SfMConfig) -> pycolmap.FeatureExtractionOptions:
    o = pycolmap.FeatureExtractionOptions()
    o.num_threads = cfg.num_threads
    o.use_gpu = cfg.use_gpu
    o.max_image_size = -1  # images are already resized by the experiment code
    if cfg.features == "sift":
        o.type = pycolmap.FeatureExtractorType.SIFT
        o.sift.max_num_features = cfg.max_num_features
    elif cfg.features in ("aliked", "aliked_lg", "aliked_nn"):
        # COLMAP only creates cameras/images here; keypoints are replaced by learned.py
        o.type = pycolmap.FeatureExtractorType.SIFT
        o.sift.max_num_features = 16
    else:
        raise ValueError(cfg.features)
    return o


def _matching_options(cfg: SfMConfig) -> pycolmap.FeatureMatchingOptions:
    o = pycolmap.FeatureMatchingOptions()
    o.num_threads = cfg.num_threads
    o.use_gpu = cfg.use_gpu
    if cfg.features == "sift":
        o.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    else:
        o.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    return o


def run_sfm(image_dir: str | Path, work_dir: str | Path, cfg: SfMConfig,
            image_names: list[str] | None = None, mask_dir: str | Path | None = None,
            overwrite: bool = True) -> SfMResult:
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    if overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    names = sorted(image_names) if image_names else sorted(
        p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    t = {}
    try:
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = cfg.camera_model
        if cfg.camera_params:
            reader.camera_params = cfg.camera_params
        if mask_dir is not None:
            reader.mask_path = str(mask_dir)
        t0 = time.time()
        pycolmap.extract_features(
            db, image_dir, image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE if cfg.single_camera else pycolmap.CameraMode.AUTO,
            reader_options=reader, extraction_options=_extraction_options(cfg),
            device=pycolmap.Device.cuda if cfg.use_gpu else pycolmap.Device.cpu)
        t["extract_s"] = time.time() - t0
        t0 = time.time()
        dev = pycolmap.Device.cuda if cfg.use_gpu else pycolmap.Device.cpu
        if cfg.features in ("aliked", "aliked_lg", "aliked_nn"):
            from .learned import aliked_lightglue_into_db
            aliked_lightglue_into_db(db, image_dir, names, work_dir, matcher=cfg.matcher, overlap=cfg.seq_overlap,
                                     matching="nn_ratio" if cfg.features == "aliked_nn" else "lightglue")
        elif cfg.matcher == "exhaustive":
            pycolmap.match_exhaustive(db, matching_options=_matching_options(cfg), device=dev)
        elif cfg.matcher == "sequential":
            po = pycolmap.SequentialPairingOptions()
            po.overlap = cfg.seq_overlap
            po.loop_detection = False
            pycolmap.match_sequential(db, matching_options=_matching_options(cfg), pairing_options=po, device=dev)
        else:
            raise ValueError(cfg.matcher)
        t["match_s"] = time.time() - t0
        t0 = time.time()
        if cfg.mapper == "incremental":
            opts = pycolmap.IncrementalPipelineOptions()
            opts.num_threads = cfg.num_threads
            opts.random_seed = cfg.seed
            if cfg.camera_params:
                opts.ba_refine_focal_length = False
                opts.ba_refine_extra_params = False
            models = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
        elif cfg.mapper == "global":
            opts = pycolmap.GlobalPipelineOptions()
            opts.num_threads = cfg.num_threads
            opts.random_seed = cfg.seed
            models = pycolmap.global_mapping(db, image_dir, sparse, options=opts)
        else:
            raise ValueError(cfg.mapper)
        t["map_s"] = time.time() - t0
        models = sorted(models.values(), key=lambda r: r.num_reg_images(), reverse=True)
        if models:
            models[0].write(sparse)  # keep the largest model at sparse/ root for convenience
        return SfMResult(ok=bool(models), n_input=len(names), models=models, timings=t)
    except Exception as e:  # pragma: no cover - failures are data points here
        return SfMResult(ok=False, n_input=len(names), timings=t, error=repr(e))
