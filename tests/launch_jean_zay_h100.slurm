#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --account=pqi@h100
#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --constraint=h100

# ________________________________________________________________________
# Minipic environment

module purge
module load arch/h100
module load nvidia-compilers/24.7
module load cmake/3.31.4
module load python

export KOKKOS_HOME="/lustre/fswork/projects/rech/oet/uwz78oa/kokkos/install_h100/"
export PATH=$KOKKOS_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KOKKOS_HOME/lib:$LD_LIBRARY_PATH

# ________________________________________________________________________
# Directory where the script was launched

cd $SLURM_SUBMIT_DIR

# ________________________________________________________________________
# Launch the test

python run.py -g kokkos_gpu --device="nvidia_h100"

