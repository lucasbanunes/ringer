#!/bin/bash
#SBATCH --job-name=mean_analysis
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00  
#SBATCH --output=/home/lbarranunes/logs/sbatch_mean_analysis.log

cd ~
singularity exec ~/imgs/ringer_base.sif bash ~/workspace/ringer/sbatch_scripts/mean_analysis.sh