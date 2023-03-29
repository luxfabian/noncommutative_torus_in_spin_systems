#!/bin/bash -x
#SBATCH --job-name=ncg_torus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=mpi-out
#SBATCH --error=mpi-err
#SBATCH --time=4:00:00
#SBATCH --partition=topo

source activate pyslurm
source settings.sh

srun python3 run.py


