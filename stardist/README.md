# StarDist for Egg Counting

## Description 

This pipeline allows for the training and evaluation of stardist models for egg counting in images,
with the removal of any objects detected that overlap with the body segmentation mask. 


## Training

Separate README.md is provided.
Code by Boris Gusev


## Evaluation

Inputs for evaluation needs to be experiment directory and filemap from pipeline setup.
Additionally, for removal of worm body in the segmentation, body segmentation masks need
to be provided. 
A standard fluorescent StarDist model will be used, and optionally XGBOOST classifier for
demarking worm segments (unnecessary if these are mapped out by body masks).

The evaluation pipeline consists of three files excluding this readme:
- run_stardist.sh
- main_script.py
- run_summary.sh

The run_stardist.sh script is needed to start the evaluation through an SBATCH job and
parse all necessary arguments to the script, while the python script contains the actual 
code for running the instance segmentaiton.
run_summary.sh can be used to generate a summary csv file of all egg counts, as normally
the counts are saved in a separate csv per worm.

### Usage
Before running make sure that the stardist_eggseg_env environment is activated. The 
necessary packages are found in stardist_env.yaml.
Then run following command through the terminal, while being in the working directory.

```
$ sbatch run_stardist.sh
```
Two output files will be created inside the working directory:
- slurm-%j.err
- slurm-%j.out

Segmentation output files will be stored inside the experiment directory given in the config file:
- stardist_count, folder containing instance segmentation masks of the eggs in the images
- stardist_seg_features, folder in the 'report' section of the experiment filemap - 
        contains the position and labels for all found eggs per worm
- stardist_report.csv, csv stored in the 'report' section providing summary of all egg counts

### Arguments
The run_stardist.sh file takes following mandatory arguments:
- EXPERIMENT_DIR: path to experiment directory as towbintools pipeline
- FILEMAP_CSV: path to experiment directory filemap
- COLUMN: column in filemap to segment, generally 'raw'
- CHANNEL: channel to segment 
- BODY_MASK_DIR: path to body segmentation mask directory
- STARDIST_SEG_OUT_DIR: path to output directory for StarDist segmentation images
- MAX_N_GPU: max number of gpus to use
- SCALE: scale at which evaluation occurs, 0.5 is recommended
- CROP_TO_SQUARE: True/False, choose to crop images
- POINT_START: choose at which worm to start segmentation
- TIME_START: choose at which time to start segmentation
- POINT_END: choose at which worm to end segmentation
- TIME_END: choose at which time to end segmentation
!! to choose open end, use -1

- STARDIST_MODEL: name of the stardist model to use, generally '2D_versatile_fluo'
- XGBOOST_MODEL: path to worm/egg classifier, e.g. 
    '/mnt/towbin.data/shared/bgusev/stardist_egg_seg/xgboost_training/egg_classifier_models/egg_classifier_v1.joblib.pkl'
