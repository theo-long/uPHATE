#!/bin/bash

#SBATCH --job-name=uphate_plots
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --mem=20480
#SBATCH -p gpu

uv run scripts/phate_uncertainty_plots.py --data=embryoid_pca --n_landmark=1000 --save --decay=15 --t=12 --knn=4