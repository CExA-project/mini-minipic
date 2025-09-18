#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --account=pqi@a100                   
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --constraint=a100

# ________________________________________________________________________
# Minipic environment

module purge
module load arch/a100
module load nvidia-compilers/24.3
module load cmake/3.31.4
module load python

export KOKKOS_HOME="/lustre/fswork/projects/rech/oet/uwz78oa/kokkos/install_a100/"
export PATH=$KOKKOS_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KOKKOS_HOME/lib:$LD_LIBRARY_PATH

# ________________________________________________________________________
# Directory where the script was launched

cd $SLURM_SUBMIT_DIR

# ________________________________________________________________________
# Launch the test

python run.py -g kokkos_gpu --device="nvidia_a100" 
#python run.py -g thrust_gpu --device="nvidia_a100" --cmake-args="-DTHRUST_ITERATOR_TYPE=counting"
#python run.py -g thrust_gpu --device="nvidia_a100" --cmake-args="-DTHRUST_ITERATOR_TYPE=zip"

