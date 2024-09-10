#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --job-name=plotting
#SBATCH --output=plotting.out
#SBATCH --error=plotting.err
#SBATCH --time=10:00:00

rm tmp_*

COL_PREFIX_DEFAULT="ch3_raw_str_"

mode=$1
input_csv=$2
output_dir=$3
col_prefix=${4:-$COL_PREFIX_DEFAULT}

python plotting.py $mode $input_csv $output_dir $col_prefix