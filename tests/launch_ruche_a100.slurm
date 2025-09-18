#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --nodes=1
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH -p gpu_test
##SBATCH -p gpu
#SBATCH -p gpua100
#SBATCH --gres=gpu:1

#module load anaconda3/2022.10/gcc-11.2.0
#export PYTHONPATH=${PWD}:$PYTHONPATH

#export NVHPC_CUDA_HOME=/gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cuda-12.2.1-6zz5mrahpuqyms5f62qdqgk6gddu3r5b

#set -x
set -eu
cd ${SLURM_SUBMIT_DIR}

# ________________________
# Kokkos

source $WORKDIR/gitlab-ci_runs/env/kokkos_a100
python run.py -g kokkos_gpu
python run.py -g kokkos_unified_gpu
python run.py -g kokkos_dualview_unified_gpu


