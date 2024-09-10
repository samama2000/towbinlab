### Channel Separator ###

# Required to automatically break down multi-channel .tiff files into their respective channels. #

# Inputs: 
    # - Input folder containing multi-channel .tiff files
    # - Output base folder
        # Within the output base folder, a separate folder for each channel will be created
    # - Optionally: channel names

# Outputs:
    # - Separate folders containing single-channel images
        # Filenames are kept


# Import
import os
import sys
import argparse
import numpy as np
import tifffile as tiff
from pathlib import Path
from random import sample
from joblib import Parallel, delayed


# Split channels and save
def separate_file(file, output_subdirs, channel_names, channel_dim):
    '''
    Takes in a tiff file f, output_subdirs [c1, c2, ...], and channel_names [c1, c2, ...], as well as channel_dim d.

        Inputs:
            file (str): string path to tiff file location
            output_dirs (list): list of string paths for the output directories of each individual channel
            channel_names (list): list of string names for each individual channel
            channel_dim (int): defines the image dimension at which the channels are separated

        Returns:
            channel_image (tiff image): single-channel tiff image saved in the corresponding folder
    '''
    # Read input file
    image = tiff.imread(file)
    
    # Check for dimensions and number of channels
    if image.ndim > 4 or image.shape[channel_dim] < len(output_subdirs):
        sys.stderr.write(f"File {file.name} does not match the expected shape [three or four dimensions] or the expected number of channels: {len(output_subdirs)}.")
        return
    
    # If channel dimension is not first dimension, move specified channel dimension to the front
    if channel_dim != 0:
        image = np.moveaxis(image, channel_dim, 0)

    # Enumerate over each channel, split the image and save
    for i, output_subdir in enumerate(output_subdirs):
        channel_image = image[i]
        output_file = Path(output_subdir) / file.name
        tiff.imwrite(output_file, channel_image)
        sys.stdout.write(f"Saved channel {channel_names[i]} to {output_file}\n")
    return


# Iterate over folders and files
def separater(input_dir, output_dir, channel_dim=0, n_files=None, channel_names=None):
    '''
    Takes in an input directory d and an output directory o, as well as an optional list of channel names [d1, d2, ...], in order to iterate over each file in the input directory
    and split it into its channels over a specified dimension c and number of files n.
    
    If no channel_names are provided, it handles the creation of the channel_names and corresponding subdirectories.
    
        Inputs:
            input_dir (str): string path to image database directory.
            output_dir (list): string path for the output base directory in which the subdirectories of each individual channel are created.
            
        Optional:
            channel_dim (int): defines the image dimension at which the channels are separated, generically 0.
            n_files (int): defines the number of images to sample from the database. If its larger than the database, then all images are split.
            channel_names (list): list of string names for each individual channel, optional as when None the function will generate channel names numerically.
            
        Returns:
            Calls the separate_file function in parallel over multiple batches of images. 
    '''

    # Check input path
    input_path = Path(input_dir)
    
    if not input_path.exists():
        sys.stderr.write(f"Input directory {input_dir} does not exist.")
        return
    
    # Get and check file list
    tiff_files = list(input_path.glob('*.tiff')) + list(input_path.glob('*.tif'))
    
    if not tiff_files:
        sys.stderr.write(f"No TIFF files found in the input directory {input_dir}.")
        return
    
    # If n_files is defined and smaller than the number of tiff files --> sample from dataset
    if n_files != None and len(tiff_files) > n_files:
        tiff_files = sample(tiff_files, n_files)
    
    # Make output directory
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        sys.stdout.write(f"Created output directory {output_dir}.")

    # Check for channel_names, if not create automatically
    if channel_names is None:
        sys.stdout.write(f"No channel names given, channel names are automatically generated database.")
        # Start counter
        max_channels = 0
        # Count channels on each file
        for file in tiff_files:
            image = tiff.imread(file)
            #Ignore wrong dimensions
            if image.ndim > 4:
                print(f"Skipping file {file.name} due to unexpected dimensions.")
                continue
            #Filter for maximum
            max_channels = max(max_channels, image.shape[channel_dim])
        # Create channel_names list
        channel_names = [f"Channel_{i+1}" for i in range(max_channels)]
    
    # Create subdirectories
    output_subdirs = [Path(output_dir) / channel_name for channel_name in channel_names]
    
    for output_subdir in output_subdirs:
        output_subdir.mkdir(parents=True, exist_ok=True)
    


    # Parallel processing
    Parallel(n_jobs=-3)(delayed(separate_file)(file, output_subdirs, channel_names, channel_dim) for file in tiff_files)

    
if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser(description="Separate image channels. This is needed for segmentation using micro-sam.")
    
    # Positional arguments
    parser.add_argument('input_dir', type=str, help='Input directory containing tiff images.')
    parser.add_argument('output_dir', type=str, help='Output directory to store results.')

    # Optional arguments
    parser.add_argument('-n', '--n_files', type=int, help='Number of files to sample from input directory.')
    parser.add_argument('-d', '--channel_dim', type=int, default = 0, help='Channel dimension from which to split (first dimension = 0).')
    parser.add_argument('-c', '--channel_names', type=str, nargs='+', help='Channel names separated by space. Example: -c red green blue.')

    args = parser.parse_args()
    
    # Use args.input_dir, args.output_dir, args.n_files, args.channel_dim, args.channel_names
    print(f"Processing {args.input_dir} to {args.output_dir}")
    if args.n_files:
        print(f"Number of files to process: {args.n_files}")
    
    print(f"Channel dimension: {args.channel_dim}")
    if args.channel_names:
        print(f"Channel names: {args.channel_names}")
    

    # Process
    separater(args.input_dir, args.output_dir, args.channel_dim, args.n_files, args.channel_names)
