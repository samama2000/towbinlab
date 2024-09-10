import os
import glob
import sys
import argparse
import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from PIL import Image
import tifffile as tiff
import numpy as np

def get_slurm_n_jobs():
    """Retrieve the number of jobs from SLURM environment variables."""
    # Try to get the number of CPUs per task from SLURM
    n_jobs = os.getenv('SLURM_CPUS_PER_TASK')
    
    if n_jobs is None:
        # If not found, try the CPUs per node environment variable
        n_jobs = os.getenv('SLURM_JOB_CPUS_PER_NODE')

    # If no SLURM environment variable is set, default to 1
    return int(n_jobs) if n_jobs else 1

def merge_bmp_to_tiff(bmp_files, output_tiff, axis=0, crop_size=None):
    """
    Merges a list of BMP images into a single multi-channel TIFF file.

    Parameters:
        bmp_files (list of str): List of file paths to the BMP images to be merged.
        output_tiff (str): File path where the output TIFF file should be saved.
    """
    # Load the BMP files and convert them to numpy arrays
    images = [np.array(Image.open(file)) for file in bmp_files]
    
    # If crop_size is provided, crop the images
    if crop_size is not None:
        width, height = crop_size
        images = [img[:height, :width] for img in images]  # Crop each image to the specified width and height


    # Stack arrays along the third dimension to create a multi-channel image
    merged_image = np.stack(images, axis=axis)  #CHANGE THE AXIS FOR THE STACK
        #important: when using multi-channel ilastik segmentation, then the channel axis has to be last (ergo, axis=2)

    # Save the merged image as a TIFF
    tiff.imwrite(output_tiff, merged_image)
    

def process_point(time_dir, point, output_dir, seq):
    """
    Processes a single point by finding, sorting, and merging associated BMP files into a TIFF file.

    Parameters:
        time_dir (str): Time directory path, where BMP images for the same time are stored.
        point (int): The specific point number to process in the time directory.
        output_dir (str): Directory where the output TIFF files should be stored.
        seq (int): Sequence number to assign to the processed file, used in naming the output file.

    Returns:
        int: Always returns 1, indicating one sequence was processed.
    """
    # Load, sort and order all BMP files from time directory
    bmp_files = glob.glob(f"{time_dir}{point}_*")
    bmp_files.sort()
    order = [1,2,0]  #CHANGE THE ORDER OF BMP FILES HERE
    bmp_files = [bmp_files[i] for i in order]

    # Load the time and point data from the directory name and the file names
    time = os.path.basename(os.path.normpath(time_dir))
    padded_time = f"{int(time):05}"
    padded_point = f"{point:04}"
    padded_seq = f"{seq:05}"

    # Generate output file name and generate TIFF file
    output_file = Path(output_dir) / f"Time{padded_time}_Point{padded_point}_470nm,565nm,BF_Seq{padded_seq}.ome.tiff"
    merge_bmp_to_tiff(bmp_files, str(output_file))
    return 1  # Return 1 to count as a processed sequence


def main(input_dir, output_dir, n_jobs=1, axis=0, crop_size=None):
    """
    Main function to orchestrate the processing of image data.

    Parameters:
        input_dir (str): Directory containing time subdirectories of BMP images.
        output_dir (str): Directory to save the processed multi-channel TIFF files.
        n_jobs (int): The number of parallel jobs to run. Do not use '-1 = all' because at least one CPU is required for file handling
    """
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print("Error: The specified directory does not exist or is not a directory.")
        exit(1)

    # Ensure the output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Start sequence counter
    seq = 0

    # Iterate over each time subdirectory in the data directory
    for time_dir in glob.glob(os.path.join(input_dir, '*/')):

        # Count the number of points in each directory
        files = os.listdir(time_dir)
        n_files = len(files)
        n_bmp = n_files - 1
        n_points = n_bmp // 3
        
        # Parallel processing of each point
        results = Parallel(n_jobs=int(n_jobs) - 2)(delayed(process_point)(time_dir, point, output_dir, seq + i) for i, point in enumerate(range(n_points)))
        seq += sum(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge BMP images into a multi-channel TIFF file with optional cropping.")
    
    parser.add_argument("input_dir", help="Directory containing the BMP files to merge", type=str)
    parser.add_argument("output_dir", help="Directory where the merged TIFF file should be saved", type=str)
    parser.add_argument("-a","--axis", default=0, type=int, help="Axis along which to stack the images (default: 0)")
    parser.add_argument("-c","--crop_size", nargs=2, type=int, help="Optional crop size as width height (e.g., --crop_size 500 500)")
    

    # Parse the arguments
    args = parser.parse_args()
    
    # If crop_size is provided, convert it to a tuple (width, height)
    crop_size = tuple(args.crop_size) if args.crop_size else None

    # Get the number of jobs from SLURM environment or default to 1
    n_jobs = get_slurm_n_jobs()

    main(args.input_dir, args.output_dir, axis=args.axis, crop_size=args.crop_size, n_jobs=n_jobs)
    