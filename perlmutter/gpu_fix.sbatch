#!/bin/bash
# Follow-up: (0) SD x4 on gerrard-hall after the tile-padding fix; (1,2) ALIKED clean runs kept on disk
# and diagnosed per image (is the low AUC a symmetry flip or a calibration problem?).
#SBATCH -J sfmrob-fix
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out
source $SCRATCH/sfmrob/code/perlmutter/env.sh
R="python -m sfmrob.run --size 1024 --threads 32"
case $SLURM_ARRAY_TASK_ID in
  0) $R --out $SFMROB_ROOT/results/raw_gpu --exp e3_superres --scene gerrard-hall --n 50 --only sr_sd_x4 --full --seeds 0 ;;
  1|2) if [ $SLURM_ARRAY_TASK_ID = 1 ]; then SC=gerrard-hall; N=50; else SC=south-building; N=40; fi
     $R --out $SFMROB_ROOT/results/diag --exp e6_methods --scene $SC --n $N --only clean_s0_f1.0_none_aliked_lg-incremental --seeds 0 --keep
     python scripts/diag_poses.py $SC $SFMROB_WORK/${SC}_${N}_1024/e6_methods/clean_s0_f1.0_none_aliked_lg-incremental_seed0/sfm/sparse ;;
esac
