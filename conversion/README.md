# Conversion for Squid

## Description 

This pipeline allows the conversion of microscope bmp-images taken by squid
into stacked tiff-images for image analysis.

The pipeline consists of three files including this readme:
- run_conversion.sh
- conversion.py
- README.md

The run_conversion.sh script is needed to start the conversion through an SBATCH job,
best utilising the potential of the HPC cluster, while the python script contains the 
actual code for the conversions and is parallelised using joblib. 

## Usage

Before running a conversion make sure that the towbintools environment is activated.
Then run following command through the terminal, while being in the conversion folder.

```
$ sbatch run_conversion.sh <input_dir> <output_dir> [options]
```

   - Options
       -h:                 Show help message
       -a, --axis:         Specify axis for image stacking
       -c, --crop_size:    Specify crop size in H W (e.g. --crop size 500 500)


If the input_dir does not exist, an error will be returned, however if the output_dir
does not exist, it will be automatically created. 

The input directory should be the raw data folder containing the individual bmp-files
in enumerated 'Time' subdirectories. The bmp-files themselves contain a number prefix
to denote the well or 'Point' of the image. Maintaining this naming convention is 
essential for correct processing of the files in the conversion. 

## Additional

-   You can change the number of CPU's used by changing '#SBATCH --cpus-per-task=XX' 
    in line 3 of the run_conversion.sh script. 
    --  The minimal number of required CPU's is 4!



