#!/bin/bash
# The GPU-only remainder, all tasks in parallel, 1 A100 each (worst case 7.25 GPU-hours if every task hits its limit).
#  0,1   SD-Turbo img2img noise (E2b), 3 seeds, gerrard / south
#  2     SD x4 upscaler (E3), seed 0, both scenes
#  3-8   ALIKED+LightGlue x {incremental, global} (E6), one task per scene x seed
#SBATCH -J sfmrob-gpu
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:50:00
#SBATCH --array=0-8
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
OUT=$SFMROB_ROOT/results/raw_gpu
R="python -m sfmrob.run --size 1024 --threads 32 --out $OUT"
G="--scene gerrard-hall --n 50"; S="--scene south-building --n 40"
T=$SLURM_ARRAY_TASK_ID
case $T in
  0) $R --exp e2_generative $G --only sdturbo_img2img --full --seeds 0 1 2 ;;
  1) $R --exp e2_generative $S --only sdturbo_img2img --full --seeds 0 1 2 ;;
  2) $R --exp e3_superres $G --only sr_sd_x4 --full --seeds 0
     $R --exp e3_superres $S --only sr_sd_x4 --full --seeds 0 ;;
  *) k=$((T-3)); SC=$([ $((k/3)) -eq 0 ] && echo "$G" || echo "$S"); SEED=$((k%3))
     $R --exp e6_methods $SC --only aliked_lg- --seeds $SEED ;;
esac
