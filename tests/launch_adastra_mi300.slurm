#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=cad15863
#SBATCH --exclusive
#SBATCH --constraint=MI300
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

export HSA_XNACK=1

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
# Modules

module purge
module load cpe/24.07
module load craype-accel-amd-gfx942 craype-x86-genoa
module load PrgEnv-cray
module load amd-mixed
module load cmake

export KOKKOS_HOME="/lus/work/CT6/cad15863/mlobet/kokkos/install_hip_mi300/"
export PATH=$KOKKOS_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KOKKOS_HOME/lib:$LD_LIBRARY_PATH

# ________________________________________________________________________
# Launch the test

python run.py -g thrust_gpu --compiler="hipcc" --device="amd_mi300" --cmake-args="-DTHRUST_ITERATOR_TYPE=counting" --backend="thrust_unified"
