#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=cin4492
#SBATCH --exclusive
#SBATCH --constraint=MI250&turbo
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
export HSA_XNACK=1
source /lus/work/CT6/cad15863/mlobet/minipic/build_recipes/env/adastra_mi250_stdpar.sh

set -x
ulimit -c unlimited
cd ${SLURM_SUBMIT_DIR}

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=THREADS
export OMP_SCHEDULE="dynamic"

echo "Date              = $(date -R) | $(date +%s)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

# ________________________________________________________________________
# Launch the test

python run.py -g thrust_gpu --compiler="hipcc" --device="amd_mi250" --cmake-args="-DTHRUST_ITERATOR_TYPE=counting" --backend="thrust_unified"