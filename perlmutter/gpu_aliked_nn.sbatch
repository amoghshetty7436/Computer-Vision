#!/bin/bash
# E6 extra arm: ALIKED features + classic mutual-NN/ratio matching (instead of LightGlue), all 8 hard
# conditions x {incremental, global}. One task per scene x seed. Isolates matcher vs features.
#SBATCH -J sfmrob-nn
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:40:00
#SBATCH --array=0-5
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
k=$SLURM_ARRAY_TASK_ID
if [ $((k/3)) -eq 0 ]; then SC="--scene gerrard-hall --n 50"; else SC="--scene south-building --n 40"; fi
python -m sfmrob.run --size 1024 --threads 32 --out $SFMROB_ROOT/results/raw_gpu --exp e6_methods $SC \
  --only aliked_nn- --seeds $((k%3))
