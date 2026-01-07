#!/bin/bash
#SBATCH --job-name=mini-minipic
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.err
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --constraint=a100

set -e

# Modules

module purge
module load "arch/a100"
module load "nvidia-compilers/24.3"
module load "cmake/3.31.4"
module load "python/3.12.7"

# Python env

if [[ -d ./venv ]]
then
    source ./venv/bin/activate
fi

# Launch the test

mini-run -g gpu-a100
