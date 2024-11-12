#!/bin/bash

#SBATCH -J eval_egg
#SBATCH -o eval_eggcount_%j.out
#SBATCH -e eval_eggcount_%j.err
#SBATCH -c 32
#SBATCH -t 72:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2

python eval_eggcount.py "$@"
