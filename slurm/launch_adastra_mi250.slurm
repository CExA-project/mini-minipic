#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --constraint=MI250&turbo
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

set -e

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
# Modules

module purge
module load cpe/24.07
module load craype-accel-amd-gfx90a craype-x86-genoa
module load PrgEnv-cray
module load amd-mixed
module load cmake
module load python

# ________________________________________________________________________
# Launch the test

mini-run -g gpu-mi250
