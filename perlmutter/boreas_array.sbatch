#!/bin/bash
# Real-weather (E1), synthetic-vs-real gap, cross-season localisation (E7) on Boreas.
#SBATCH -J sfmrob-boreas
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 01:30:00
#SBATCH --array=0-7
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
# every task does its shard of all three modes (keeps the charge estimate small)
N=8
SH=$SLURM_ARRAY_TASK_ID
for MODE in weather xseason synth; do
  SEEDS="0 1 2"; [ $MODE = synth ] && SEEDS="0"; [ $MODE = xseason ] && SEEDS="0 1"
  python scripts/boreas_experiments.py --mode $MODE --seeds $SEEDS --shard $SH/$N --threads 16 \
    --out $SFMROB_ROOT/results/raw --work $SFMROB_WORK
done
