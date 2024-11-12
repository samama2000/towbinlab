#!/bin/bash

#SBATCH -J train_egg
#SBATCH -o tr_eggcount.out
#SBATCH -e tr_eggcount.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2

python train_eggcount.py "$@"
