# Source me. Everything lives on $SCRATCH (fast Lustre, purged after 8 weeks).
export SFMROB_ROOT=${SFMROB_ROOT:-$SCRATCH/sfmrob}
export SFMROB_DATA=$SFMROB_ROOT/data
export SFMROB_WORK=$SFMROB_ROOT/work
export HF_HOME=$SFMROB_ROOT/hf
export TORCH_HOME=$SFMROB_ROOT/torch
export GLOG_minloglevel=2
export PYTHONPATH=$SFMROB_ROOT/code:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
module load python 2>/dev/null
set +u; source $SFMROB_ROOT/venv/bin/activate; set -u 2>/dev/null || true
cd $SFMROB_ROOT/code
