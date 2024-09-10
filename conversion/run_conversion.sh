#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --job-name=conversion
#SBATCH --output=conversion.out
#SBATCH --error=conversion.err
#SBATCH --time=1-00:00:00

# Pass command line arguments to the python script

python conversion.py "$@"

# Usage: sbatch run_conversion.sh {input_dir} {output_dir}

# ! ! ! Minimum number of CPUs per task = 4 ! ! !

# Make sure to have towbintools activated before running

# The input directory should contain time subdirectories that are named after the hour of the image capture, and within
# are the corresponding BMP files separated by points through their names in the following way. 
#           input_dir
#           |
#           |_ 0
#              |_ 0_0_0_0_brightfield
#              |_ 0_0_0_0_body
#              |_ 0_0_0_0_germline
#              |_ 1_0_0_0_brightfield
#              |_ 1_0_0_0_body
#              |_ 1_0_0_0_germline
#              |_ 2_0_0_0_brightfield
#              |_ 2_0_0_0_brightfield
#              |_ ...
#           |_ 1
#           |_ 2
#           |_ 3
#           |_ 4
#           |_ 5
#           |_ ... 
#
# The number of images to merge into a single TIFF does not matter, 
# as long as they have the same digit as the prefix <digit>_*
