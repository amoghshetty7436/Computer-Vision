#!/bin/bash
# GPU image synthesis only (Real-ESRGAN, LaMa, SD-Turbo img2img, SD x4 upscaler): 1 A100 per scene.
#SBATCH -J sfmrob-gen
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
SCENES=(gerrard-hall south-building person-hall)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
for EXP in e2_generative e3_superres e4_crowds e5_contamination; do
  python -m sfmrob.run --exp $EXP --scene $SCENE --n 40 --size 1280 --full --seeds 0 1 2 \
    --phase gen --out $SFMROB_ROOT/results/raw
done
