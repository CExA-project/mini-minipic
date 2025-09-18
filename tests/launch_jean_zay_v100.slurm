#!/bin/bash
#SBATCH --job-name=minipic            # Job name
#SBATCH -A oet@v100                   # If relevant (multiple projects), accounts eligible for the user
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=1                   # Number of MPI processes (= total number of GPU)
##SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par nœud (max 8 avec gpu_p2)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:20:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=output        # Name of the output file
#SBATCH --error=error         # Name of the error file

# Minipic environment
source ~/env/minipic.sh

# echo des commandes lancees
set -x

# Directory where the script was launched
cd $SLURM_SUBMIT_DIR

# ________________________
# Kokkos

python run.py -g kokkos_gpu
python run.py -g kokkos_unified_gpu
python run.py -g kokkos_dualview_unified_gpu


