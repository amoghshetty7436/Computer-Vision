#!/bin/bash
# One-time setup on a Perlmutter LOGIN node:  bash perlmutter/setup.sh
# Creates venv, downloads datasets + model weights, prepares the image subsets
# (so the parallel array tasks never race on downloads).
set -euo pipefail
export SFMROB_ROOT=${SFMROB_ROOT:-$SCRATCH/sfmrob}
mkdir -p $SFMROB_ROOT/{data,work,hf,torch,logs}
module load python
if [ ! -d $SFMROB_ROOT/venv ]; then
  python -m venv $SFMROB_ROOT/venv
  source $SFMROB_ROOT/venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -r $SFMROB_ROOT/code/requirements.txt
fi
source $SFMROB_ROOT/code/perlmutter/env.sh
python - <<'PY'
from pathlib import Path
import os, urllib.request
from sfmrob import data as D
from sfmrob.perturb import generative as G
from sfmrob.perturb.crowds import person_bank
for s in ["gerrard-hall", "south-building", "person-hall"]:
    D.fetch_colmap_scene(s)
    D.prepare_colmap_subset(s, 60, 1600, Path(os.environ["SFMROB_WORK"]) / f"{s}_60_1600" / "clean")
    print("prepared", s, flush=True)
person_bank()
G._weights("RealESRGAN_x4plus.pth"); G._weights("big-lama.pt")
ck = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth", ck / "aliked-n16.pth")
import kornia.feature as KF; KF.LightGlue("aliked")
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sd-turbo", allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])
snapshot_download("stabilityai/stable-diffusion-x4-upscaler", allow_patterns=["*.json", "*.txt", "*.safetensors"])
print("setup done")
PY
