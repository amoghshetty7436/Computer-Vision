#!/bin/bash
# Controlled experiments (E1s,E2,E3,E4,E5) on CPU nodes: 3 scenes x 16 shards = 48 tasks.
#SBATCH -J sfmrob-cpu
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 03:00:00
#SBATCH --array=0-47
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
SCENES=(gerrard-hall south-building person-hall)
NSH=16
SCENE=${SCENES[$((SLURM_ARRAY_TASK_ID / NSH))]}
SH=$((SLURM_ARRAY_TASK_ID % NSH))
PHASE=${PHASE:-cpu}
for EXP in e2_analytic e1_synth_weather e4_crowds e3_superres e2_generative e5_contamination; do
  python -m sfmrob.run --exp $EXP --scene $SCENE --n 60 --size 1600 --full --seeds 0 1 2 \
    --phase $PHASE --shard $SH/$NSH --threads 16 --out $SFMROB_ROOT/results/raw
done
