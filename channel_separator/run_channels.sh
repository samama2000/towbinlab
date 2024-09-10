#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --job-name=channel_sep
#SBATCH --output=channels.out
#SBATCH --error=channels.err
#SBATCH --time=1-00:00:00

# Pass command line arguments to the SBATCH script
python channels.py "$@"
