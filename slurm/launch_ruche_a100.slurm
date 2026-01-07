#!/bin/bash
#SBATCH --job-name=mini-minipic
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.err
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH -p gpu_test
##SBATCH -p gpu
#SBATCH -p gpua100
#SBATCH --gres=gpu:1

set -e

# Modules

module purge
module load "gcc/13.2.0/gcc-4.8.5"
module load "cuda/12.8.0/gcc-13.2.0"
module load "cmake/3.28.3/gcc-11.2.0"
module load "python-nospack/3.12.4/gcc-11.2.0"

# Python env

if [[ -d ./venv ]]
then
    source ./venv/bin/activate
fi

# Launch the test

mini-run -g gpu-a100 -s thermal_heavy
