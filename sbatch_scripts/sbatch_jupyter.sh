#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cpu

# Usage:
# $sbatch sbatch_jupyter.sh <image-name> <port-number>
# $sbatch -o $HOME/logs/sbatch_jupyter.log sbatch_jupyter.sh ringer.sif 1234

cd ~
singularity run --env jupyter_port=$2 imgs/$1