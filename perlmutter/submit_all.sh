#!/bin/bash
# Usage (login node):  ACCOUNT=mXXXX bash perlmutter/submit_all.sh
# GPU budget: gen 3 x 1.25h + e6 4 x 0.83h = 7.1 A100-hours *maximum* (time limits are hard caps).
set -euo pipefail
: "${ACCOUNT:?set ACCOUNT=mXXXX}"
source $SCRATCH/sfmrob/code/perlmutter/env.sh
mkdir -p logs $SFMROB_ROOT/results/raw
A="-A $ACCOUNT"
GEN=$(sbatch --parsable $A perlmutter/gpu_gen.sbatch)
CPU1=$(sbatch --parsable $A perlmutter/cpu_array.sbatch)
# second CPU pass picks up the conditions whose images the GPU job synthesised
CPU2=$(sbatch --parsable $A --dependency=afterany:$GEN --export=ALL,PHASE=cpu -J sfmrob-cpu2 perlmutter/cpu_array.sbatch)
E6G=$(sbatch --parsable $A perlmutter/gpu_e6.sbatch)
E6C=$(sbatch --parsable $A --dependency=afterany:$GEN perlmutter/cpu_e6.sbatch)
BOR=$(sbatch --parsable $A perlmutter/boreas_array.sbatch)
echo "gen=$GEN cpu1=$CPU1 cpu2=$CPU2 e6gpu=$E6G e6cpu=$E6C boreas=$BOR" | tee $SFMROB_ROOT/jobs.txt
