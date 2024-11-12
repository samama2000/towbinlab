# CSRNet for Egg Counting

## Description 

This pipeline allows for the training and evaluation of CSRNet models for egg counting in images.


## Training

Inputs for training need to be .json annotation files from 'LabelMe', containing point
annotations of eggs. Important: In 'LabelMe' save the image data inside of the json - 
as the training model will load it from there.

The training pipeline consists of three files excluding this readme:
- tr_config.yaml
- run_tr_eggcount.sh
- train_eggcount.py


The run_tr_eggcount.sh script is needed to start the training through an SBATCH job,
best utilising the potential of the HPC cluster, while the python script contains the 
actual code for training, while the tr_config.yaml file parses all necessary arguments
to the script.

### Usage
Before running make sure that the towbintools environment is activated.
Then run following command through the terminal, while being in the working directory.

```
$ sbatch run_tr_config.sh tr_config.yaml
```
Two output files will be created inside the working directory:
- tr_eggcount.err
- tr_eggcount.out

Output files will be created inside the save directory given in the config file:
- training.log, contains information on training such as precision, recall and F1
- model_epoch_XX_f1_YY.pth, best model from epoch XX with an F1 score of YY

### Arguments
The training config file takes following mandatory arguments:
- json_directory - path to the .json annotation files
- save_dir - save directory
- label - string of label used in 'LabelMe' annotation, e.g. "egg"
- pretrained - True or False for use of pretrained VGG16 layer
- num_epochs
- batch_size
- learning_rate

Additionally, following optional arguments can be used:
- log_file - give a specific name to the log file - else "training.log"
- use_gpu - change to True if gpus available - else False
- gpu_device - give a list of range(num_gpus) - else None
- transform - give a normalisation transform to use for the data - else None
- tile_size - single integer value for quadratic tile size - else None
- overlap - single integer value for overlap - else 0
- threshold - threshold value for binary mask - else 0.5
- min_size - minimal size for small object removal - else 20
- min_distance - minimal distance for local peak detection - else 25
- match_distance - minimal distance for centroid matching - else 25
- tile_dir - string path for optional saving tile inspections every 20 batches - else None

## Evaluation

Inputs for evaluation needs to be TIFF file images to count eggs on and the trained CSRNet model.

The evaluation pipeline consists of three files excluding this readme:
- eval_config.yaml
- run_eval_eggcount.sh
- eval_eggcount.py


The run_eval_eggcount.sh script is needed to start the evaluation through an SBATCH job,
while the python script contains the actual code for training and the eval_config.yaml 
file parses all necessary arguments to the script.

### Usage
Before running make sure that the towbintools environment is activated.
Then run following command through the terminal, while being in the working directory.

```
$ sbatch run_eval_config.sh eval_config.yaml
```
Two output files will be created inside the working directory:
- eval_eggcount_%j.err
- eval_eggcount_%j.out

Two output files will be created inside the save directory given in the config file:
- eggcount.log, contains information on training such as precision, recall and F1
- results.csv, contains egg counting result in CSV format

### Arguments
The training config file takes following mandatory arguments:
- image_dir - path to the input image directory
- output_dir - path to the output directory
- model_path - path to the trained CSRNet model

All other arguments are optional:
- channels - select a channel from the image to process - else [0]
- augment_contrast - use augment_contrast from pipeline - else False
- log_file - give a specific name to the log file - else "eggcount.log"
- use_gpu - change to True if gpus available - else False
- gpu_device - give a list of range(num_gpus) - else None
- batch_size - batch size for evaluation - else 4
- num_workers - number of workers for dataset - else 16
- tile_size - single integer value for quadratic tile size - else None
- overlap - single integer value for overlap - else 0
- threshold - threshold value for binary mask - else 0.5
- min_size - minimal size for small object removal - else 20
- min_distance - minimal distance for local peak detection - else 25
- create_json - saves egg counts as annotation .json files for further training - else None
- save_visual - saves results as .png images of watershed egg segmentation - else None


Important: Visualising the results is computationally expensive. 
