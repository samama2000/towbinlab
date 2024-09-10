# Channel Separator for Napari Image Analysis

## Description 

This pipeline allows for the effective splitting of images into their respective channels.
Images inputs and outputs NEED to be in .tif or .tiff format.

The pipeline consists of two files excluding this readme:
- run_channels.sh
- channels.py


The run_channel.sh script is needed to start the conversion through an SBATCH job,
best utilising the potential of the HPC cluster, while the python script contains the 
actual code for splitting and is parallelised using joblib.


## Usage

Before running a conversion make sure that the towbintools environment is activated.
Then run following command through the terminal, while being in the channels folder.

```
$ sbatch run_channels.sh [opt] <input_dir> <output_dir>
```

If the input_dir does not exist, an error will be returned, however if the output_dir
does not exist, it will be automatically created. 

The input directory should be the folder containing the image tiff-files. 
The tiff-images are subsequently split and each individual channel is saved into a 
respective 'channel subdirectory' within the output directory.

### Optional Arguments

Next to the two positional arguments (input and output directory) we can also include some
optional arguments. You can add them before or after the stationary arguments, it is only
important to use the right tag.

e.g. $ sbatch run_channels.py -n 1000 /input_dir /output_dir
    -- This would make it so that the script samples and splits only 1000 images.

Here are all possible tags:

    -n, --n_files:          (int) Define number of files to sample from input directory. Useful for creating training datasets.
    -d, --channel_dim:      (int) Define channel dimension from which to split. First dimension = 0.
    -c, --channel_names:    (str) Define channel names. Separate using space, e.g. red green blue.
    -h:                           Print help and exit.

If no optional arguments are given, the script will assume that all files in the input directory
need to be split. The channel dimension is set to 0 as default and the channel names are generated
automatically by enumerating over the channels in the database. 

To note: If less channel names are given than there are channels, then only named channels will
be split and the others ignored. For example, if I define -c 'red blue' for three-channel tiff 
images, then only the first two channels will be split and their corresponding subdirectories 
will be called 'red' and 'blue'. However, if more channel names are given than there are channels,
an error will be returned by the script.

## Additional

-   You can change the number of CPU's used by changing '#SBATCH --cpus-per-task=XX' 
    in line 3 of the run_channels.sh script. 
    --  The minimal number of required CPU's is 4!

    

