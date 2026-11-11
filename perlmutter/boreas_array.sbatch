#!/bin/bash
# Real-weather (E1), synthetic-vs-real gap, cross-season localisation (E7) on Boreas.
#SBATCH -J sfmrob-boreas
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 03:00:00
#SBATCH --array=0-23
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
N=8
MODES=(weather xseason synth)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID / N))]}
SH=$((SLURM_ARRAY_TASK_ID % N))
SEEDS="0 1 2"; [ $MODE = synth ] && SEEDS="0"
python scripts/boreas_experiments.py --mode $MODE --seeds $SEEDS --shard $SH/$N --threads 16 \
  --out $SFMROB_ROOT/results/raw --work $SFMROB_WORK
