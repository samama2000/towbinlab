#!/bin/bash

#SBATCH --job-name="stardist_submit"
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:2

python train.py "$@"