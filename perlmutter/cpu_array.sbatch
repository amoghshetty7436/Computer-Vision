#!/bin/bash
# Controlled experiments (E1s,E2,E3,E4,E5 + SIFT half of E6) on CPU: 3 scenes x NSH shards.
# -c 16 = 8 physical cores = 1/16 node. Charge estimate = tasks x time / 16 node-hours.
#SBATCH -J sfmrob-cpu
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 01:30:00
#SBATCH --array=0-23
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
SCENES=(gerrard-hall south-building person-hall)
NSH=${NSH:-8}
SCENE=${SCENES[$((SLURM_ARRAY_TASK_ID / NSH))]}
SH=$((SLURM_ARRAY_TASK_ID % NSH))
for EXP in e2_analytic e1_synth_weather e4_crowds e3_superres e2_generative e5_contamination e6_methods; do
  python -m sfmrob.run --exp $EXP --scene $SCENE --n 40 --size 1280 --full --seeds 0 1 2 \
    --phase cpu --shard $SH/$NSH --threads 16 --out $SFMROB_ROOT/results/raw
done
