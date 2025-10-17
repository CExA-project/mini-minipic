#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --nodes=1
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_short

set -e

# ________________________________________________________________________
# Modules

module purge
module load "gcc/13.2.0/gcc-4.8.5"
module load "cmake/3.21.4/gcc-13.2.0"

# ________________________________________________________________________
# Launch the test

mini-run -g cpu-openmp
