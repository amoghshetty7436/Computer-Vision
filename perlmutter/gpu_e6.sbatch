#!/bin/bash
# E6: ALIKED+LightGlue (GPU matching) with incremental and global mapping. 4 shards.
#SBATCH -J sfmrob-e6
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:40:00
#SBATCH --array=0-3
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
for SCENE in gerrard-hall south-building person-hall; do
  python -m sfmrob.run --exp e6_methods --scene $SCENE --n 40 --size 1280 --seeds 0 \
    --phase gpu_sfm --shard $SLURM_ARRAY_TASK_ID/4 --threads 32 --out $SFMROB_ROOT/results/raw
done
