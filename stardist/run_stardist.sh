#!/bin/bash

#SBATCH --job-name="stardist_submit"
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1


# if there is an error, stop the script rather than continuing past the error and potentially doing something unexpected
set -eu

##########
# CONFIG #
##########

# filemap configuration

# EXPERIMENT_DIR only used to automatically define the filemap csv the output directory
# you can otherwise manually specify FILEMAP_CSV or STARDIST_SEG_OUT_DIR
EXPERIMENT_DIR="/mnt/towbin.data/shared/smarin/analysis/lifespan_20240404-20240605"

FILEMAP_CSV="$EXPERIMENT_DIR/analysis/report/analysis_filemap.csv"
# which column within filemap to segment. Usually 'raw', but might be 'analysis/...'
COLUMN='raw'
# 0-indexed image channel in experiment/COLUMN/<image_name>.tiff; depends on how the channels are arranged within the raw/image_file.tiff
# if file is arranged as expected, CHANNEL=1 will be microscopy-ch2
CHANNEL=1

BODY_MASK_DIR="$EXPERIMENT_DIR/analysis/ch3_seg_body"
STARDIST_SEG_OUT_DIR="$EXPERIMENT_DIR/analysis/stardist_count"

# running options

# will use at most MAX_N_GPU gpus. Could be less if some gpus are already in use by others. Does not discriminate between gpu types
MAX_N_GPU=2

# Set the scale for images for the GPU step. 1 means no scaling, 0.5 means half the size in each dimension (i.e. 1/4 of all pixels), etc.
# Smaller is faster but technically less accurate. In my experience 0.5 is still plenty accurate.
SCALE=0.5

# should rectangular images be cropped to a centered square before running stardist?
# if rectangular images show neighbouring wells, and the well of interest is in the centre, cropping to square will filter out the neighbouring wells
CROP_TO_SQUARE=False

# both at 0 to start from Time0000_Point0000
POINT_START=0
TIME_START=0

# -1 to process all points, or all times. upper bound is inclusive - POINT_END=5 and TIME_END=10 will process all points/times up to and including Time0010_Point0005
# upper bound is inclusive because slurm job arrays are upper bound inclusive, unlike python
POINT_END=97
TIME_END=-1



# model configuration

# custom stardist models not yet supported. Choice of '2D_versatile_fluo', '2D_versatile_he', or '2D_paper_dsb2018'.
# See https://github.com/stardist/stardist#pretrained-models-for-2d
STARDIST_MODEL='2D_versatile_fluo'
XGBOOST_MODEL='/mnt/towbin.data/shared/bgusev/stardist_egg_seg/xgboost_training/egg_classifier_models/egg_classifier_v1.joblib.pkl'


##############
# CONFIG END #
##############

function ensure_conda_env() {
    env_name=$1
    env_file=$2

    # Check if the environment exists
    if [[ $(micromamba env list | grep -w $env_name) ]]; then
        echo "Environment $env_name already exists. Continuing..."
    else
        echo "Environment $env_name does not exist. Creating..."
        # need to request gpu so that CUDA is available in the environment when resolving dependencies
        micromamba env create -f $env_file -y
        echo "Environment $env_name created."
    fi
}

ensure_conda_env 'stardist_eggseg_env' '/mnt/towbin.data/shared/bgusev/stardist_egg_seg/stardist_frozen_env.yaml'


# To process the data quicker, we want to use multiple GPUs. However, stardist was not written with multiple GPUs in mind.
# So, to run using multiple GPUs, I use a slurm job-array, with the array index corresponding to the 'Point' for the python script to process.
# However, first need to know how big the job-array needs to be to process all points.

# Handle if want to process all points
if [ "$POINT_END" -eq -1 ]; then
    # use awk to read csv and find the max point. awk script generated with github copilot
    # Name of the column
    COLUMN_NAME="Point"
    # Get column index
    COLUMN_INDEX=$(awk -v column="$COLUMN_NAME" -F, 'NR==1 {for (i=1; i<=NF; i++) {if ($i == column) {print i; exit}}} ' $FILEMAP_CSV)
    # Find max value
    POINT_END=$(awk -v column="$COLUMN_INDEX" -F, 'NR>1 {if ($column > max) max=$column} END {print max+0}' $FILEMAP_CSV)
fi


if [[ "$POINT_END" =~ ^[0-9]+$ ]]
then
    echo Processing points from $POINT_START up to and including $POINT_END
else
    echo "POINT_END is not a positive integer."
    echo $POINT_END
    exit 1
fi

mkdir -p $STARDIST_SEG_OUT_DIR

err_out_dir=STARDIST_JOB_OUPUT-"$SLURM_JOB_ID"
mkdir "$err_out_dir"
# --parsable returns the job id of the array job that is then captured by JOB_ID variable
JOB_ID=$(sbatch \
    --parsable \
    --job-name='stardist_seg' \
    --cpus-per-task=16 \
    --time='3:00:00' \
    --mem='24GB' \
    --output="$err_out_dir/Point-%a.out" \
    --error="$err_out_dir/Point-%a.err" \
    --array="$POINT_START-$POINT_END%$MAX_N_GPU" \
    --gres='gpu:1' \
    --wrap="~/.local/bin/micromamba run -n stardist_eggseg_env python3 main_script.py --filemap $FILEMAP_CSV --out_dir $STARDIST_SEG_OUT_DIR --scale $SCALE --crop $CROP_TO_SQUARE --t_start $TIME_START --t_end $TIME_END --channel $CHANNEL --column $COLUMN --xgboost $XGBOOST_MODEL --stardist $STARDIST_MODEL --body_mask_dir $BODY_MASK_DIR" \
)



summary_script=$(cat <<END
import pandas as pd
from pathlib import Path

csv_dir = Path('$FILEMAP_CSV').parent / 'stardist_seg_features'
csv_files = list(csv_dir.glob('*.csv'))

summary_file = csv_dir.parent / 'stardist_report.csv'
print(summary_file)
summary_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
summary_df = (
    summary_df
        .groupby(['Time', 'Point'])
        .size()
        .reset_index(name='egg-count')
        .sort_values(by=['Point', 'Time'])
)
summary_df.to_csv(summary_file, index=False)

END
)
# "$summary_script" needs to be in "" quotes, else the whitespace is stripped, which python syntax needs
# Counterintuitevly, 'afterany' clause is satisfied after all tasks in the job array complete.
# I think the semantics are 'after all jobs finish and *any* successful'
sbatch \
    --dependency=afterany:$JOB_ID \
    --job-name='sdreport' \
    --cpus-per-task=4 \
    --time='1:00:00' \
    --mem='32GB' \
    --output="$err_out_dir/_summary_job.out" \
    --error="$err_out_dir/_summary_job.err" \
    --wrap="~/.local/bin/micromamba run -n stardist_eggseg_env python3 -c \"$summary_script\"" \


    