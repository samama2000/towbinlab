#!/bin/bash
#SBATCH --output=tmp_channels.out
#SBATCH --error=tmp_channels.err

# Pass command line arguments to the SBATCH script
sbatch _sbatch_plotting.sh "$@"
