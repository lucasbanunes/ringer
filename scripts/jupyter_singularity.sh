#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cpu

# Usage:
# $sbatch jupyter_singularity.sh <image-name> <setup-script> <jupyter-port>
# $sbatch -o $HOME/logs/jupyter_singularity.log jupyter_singularity.sh root-cern_latest.sif setup_repos_kepler.sh 1234
# $sbatch jupyter_singularity.sh root-cern_latest.sif setup_repos_kepler.sh 1234
# $sbatch jupyter_singularity.sh root-cern_latest.sif setup_repos.sh 1234

singularity exec $HOME/imgs/$1 bash launch_jupyter.sh $2 $3