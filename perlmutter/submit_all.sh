#!/bin/bash
# Usage (login node, from $SCRATCH/sfmrob/code):  ACCOUNT=mXXXX bash perlmutter/submit_all.sh
# Submits setup + all experiments with dependencies; no live login needed afterwards.
# Budgets (Slurm charges/estimates by requested time; -c 16 shared = 1/16 CPU node):
#   CPU: cpu1 24x1.5h/16 = 2.25 + cpu2 12x1h/16 = 0.75 + boreas 8x1.5h/16 = 0.75 + setup 0.06  = 3.8 node-h
#   GPU: gen 3 x 1.0h + e6 4 x 0.67h = 5.7 A100-hours maximum
set -euo pipefail
: "${ACCOUNT:?set ACCOUNT=mXXXX}"
cd $SCRATCH/sfmrob/code
mkdir -p logs $SCRATCH/sfmrob/results/raw
A="-A $ACCOUNT"
if [ -f $SCRATCH/sfmrob/.setup_ok ]; then DEP=""; else
  SETUP=$(sbatch --parsable $A perlmutter/setup.sbatch); DEP="--dependency=afterok:$SETUP"; fi
GEN=$(sbatch --parsable $A $DEP perlmutter/gpu_gen.sbatch)
CPU1=$(sbatch --parsable $A $DEP --export=ALL,NSH=8 perlmutter/cpu_array.sbatch)
# second, smaller CPU pass: conditions whose inputs the GPU job synthesised
CPU2=$(sbatch --parsable $A --dependency=afterany:$GEN:$CPU1 --export=ALL,NSH=4 --array=0-11 -t 01:00:00 \
  -J sfmrob-cpu2 perlmutter/cpu_array.sbatch)
E6G=$(sbatch --parsable $A $DEP perlmutter/gpu_e6.sbatch)
BOR=$(sbatch --parsable $A $DEP perlmutter/boreas_array.sbatch)
echo "setup=${SETUP:-none} gen=$GEN cpu1=$CPU1 cpu2=$CPU2 e6gpu=$E6G boreas=$BOR" | tee $SCRATCH/sfmrob/jobs.txt
