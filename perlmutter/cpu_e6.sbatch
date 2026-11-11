#!/bin/bash
# E6 SIFT half (incremental vs global) on CPU; GPU-generated inputs come from gpu_gen.
#SBATCH -J sfmrob-e6cpu
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 02:00:00
#SBATCH --array=0-5
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
SCENES=(gerrard-hall south-building person-hall)
SCENE=${SCENES[$((SLURM_ARRAY_TASK_ID / 2))]}
python -m sfmrob.run --exp e6_methods --scene $SCENE --n 60 --size 1600 --seeds 0 1 2 \
  --phase cpu --shard $((SLURM_ARRAY_TASK_ID % 2))/2 --threads 16 --out $SFMROB_ROOT/results/raw
