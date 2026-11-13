#!/bin/bash
# Setup for the GPU-only remainder (run on a LOGIN node): bash perlmutter/setup_gpu.sh
# Same scene subsets as the finished CPU runs: gerrard-hall 50 images, south-building 40 images, 1024 px.
set -eo pipefail
source $SCRATCH/sfmrob/code/perlmutter/env.sh
python - <<'PY'
from pathlib import Path
import os, urllib.request
from sfmrob import data as D
from sfmrob.perturb.crowds import person_bank
for s, n in [("gerrard-hall", 50), ("south-building", 40)]:
    D.prepare_colmap_subset(s, n, 1024, Path(os.environ["SFMROB_WORK"]) / f"{s}_{n}_1024" / "clean")
    print("prepared", s, flush=True)
person_bank()
ck = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
if not (ck / "aliked-n16.pth").exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth", ck / "aliked-n16.pth")
import kornia.feature as KF; KF.LightGlue("aliked")
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sd-turbo", allow_patterns=["*.json", "*.txt", "*.safetensors"])
snapshot_download("stabilityai/stable-diffusion-x4-upscaler", allow_patterns=["*.json", "*.txt", "*.safetensors"])
open(os.environ["SFMROB_ROOT"] + "/.setup_gpu_ok", "w").write("ok")
print("setup_gpu done")
PY
