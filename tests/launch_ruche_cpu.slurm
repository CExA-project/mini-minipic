#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --nodes=1
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_short

set -x
cd ${SLURM_SUBMIT_DIR}

# _________________________________________
# OpenMP g++

source ${HOME}/env/kokkos_omp_cpu
python run.py -c g++ 

# _________________________________________
# OpenMP icpx

source ${WORKDIR}/gitlab-ci_runs/env/kokkos_omp_cpu_intel
python run.py -c icpx

# _________________________________________
# Kokkos icpx

source ${WORKDIR}/gitlab-ci_runs/env/kokkos_omp_cpu_intel
python run.py -g kokkos -c icpx
#python run.py -g kokkos_unified -c icpx

