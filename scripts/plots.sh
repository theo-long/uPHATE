#!/bin/bash

#SBATCH --job-name=uphate_plots
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=h200:1
#SBATCH --mem=40960
#SBATCH -p gpu_h200

uv run scripts/phate_uncertainty_plots.py --data=embryoid_pca --n_landmark=1000 --save --decay=15 --t=12 --knn=4