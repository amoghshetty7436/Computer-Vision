#!/bin/bash
# Runs perlmutter/setup.sh on a compute node (venv, datasets, weights, image subsets).
#SBATCH -J sfmrob-setup
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
bash $SCRATCH/sfmrob/code/perlmutter/setup.sh
