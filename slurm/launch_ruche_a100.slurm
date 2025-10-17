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

set -e

# ________________________________________________________________________
# Modules

module purge
module load "gcc/13.2.0/gcc-4.8.5"
module load "cuda/12.8.0/gcc-13.2.0"
module load "cmake/3.21.4/gcc-13.2.0"

# ________________________________________________________________________
# Launch the test

mini-run -g gpu-a100
