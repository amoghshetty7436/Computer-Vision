#!/bin/bash
# Usage (login node, from $SCRATCH/sfmrob/code):  ACCOUNT=mXXXX bash perlmutter/submit_all.sh
# Submits setup + all experiments with dependencies; nothing needs a live login after this.
# GPU budget: gen 3 x 1.25h + e6 4 x 0.83h = 7.1 A100-hours *maximum* (time limits are hard caps).
set -euo pipefail
: "${ACCOUNT:?set ACCOUNT=mXXXX}"
cd $SCRATCH/sfmrob/code
mkdir -p logs $SCRATCH/sfmrob/results/raw
A="-A $ACCOUNT"
if [ -f $SCRATCH/sfmrob/.setup_ok ]; then DEP=""; else
  SETUP=$(sbatch --parsable $A perlmutter/setup.sbatch); DEP="--dependency=afterok:$SETUP"; fi
GEN=$(sbatch --parsable $A $DEP perlmutter/gpu_gen.sbatch)
CPU1=$(sbatch --parsable $A $DEP perlmutter/cpu_array.sbatch)
CPU2=$(sbatch --parsable $A --dependency=afterany:$GEN:$CPU1 -J sfmrob-cpu2 perlmutter/cpu_array.sbatch)
E6G=$(sbatch --parsable $A $DEP perlmutter/gpu_e6.sbatch)
E6C=$(sbatch --parsable $A --dependency=afterany:$GEN perlmutter/cpu_e6.sbatch)
BOR=$(sbatch --parsable $A $DEP perlmutter/boreas_array.sbatch)
echo "setup=${SETUP:-none} gen=$GEN cpu1=$CPU1 cpu2=$CPU2 e6gpu=$E6G e6cpu=$E6C boreas=$BOR" | tee $SCRATCH/sfmrob/jobs.txt
