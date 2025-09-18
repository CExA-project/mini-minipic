#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --nodes=1
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
##SBATCH -p gpu_test
#SBATCH -p gpu
##SBATCH -p gpua100
#SBATCH --gres=gpu:1

set -x
cd ${SLURM_SUBMIT_DIR}

export PYTHONPATH=${PWD}:$PYTHONPATH

# ________________________
# Kokkos

source $WORKDIR/gitlab-ci_runs/env/kokkos_v100
python run.py -g kokkos_gpu
#python run.py -g kokkos_unified_gpu
#python run.py -g kokkos_dualview_unified_gpu

