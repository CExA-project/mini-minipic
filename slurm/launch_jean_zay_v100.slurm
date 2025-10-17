#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=output
#SBATCH --error=error

set -e

# ________________________________________________________________________
# Modules

module purge
module load nvidia-compilers/24.7
module load cmake/3.31.4
module load python

# ________________________________________________________________________
# Launch the test

python run.py -g gpu-v100
