import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from joblib import delayed, Parallel
import re
from PIL import Image, ImageOps

### CSV_DATA ###

def plot_worm_data(csv_file, output_folder, col_prefix='ch3_raw_str_'):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Filter columns to get only those containing the measurements
    measurement_cols = [col for col in data.columns if col.startswith(col_prefix)]
    
    # Iterate over each unique worm identified by 'Point'
    for point in data['Point'].unique():
        # Filter data for the current worm
        worm_data = data[data['Point'] == point]
        
        # Create a figure and axes for the plots
        fig, axs = plt.subplots(len(measurement_cols), 1, figsize=(10, 15), sharex=True)
        
        # Plot each measurement
        for ax, col in zip(axs, measurement_cols):
            ax.plot(worm_data['Time'], worm_data[col], marker='o', linestyle='-', color='blue')
            ax.set_title('{} over Time for Worm {}'.format(col, point))
            ax.set_xlabel('Time')
            ax.set_ylabel(col.split('_')[-1])  # Assumes last part of column name is the descriptor
        
        # Tight layout to handle overlapping
        plt.tight_layout()
        
        # Save the figure as an SVG file
        fig.savefig(os.path.join(output_folder,'{}.svg').format(point))
        
        # Close the plot to free memory
        plt.close(fig)


### IMAGE_DATA ###

def resize_image(img, resize_param):
    """Resize image based on the provided parameter.
    
    Args:
        img (PIL.Image.Image): The image to resize.
        resize_param (float or tuple): Scalar for proportional resizing or (width, height) for exact resizing.

    Returns:
        PIL.Image.Image: Resized image.
    """
    if isinstance(resize_param, tuple):
        # Resize to exact dimensions
        return img.resize(resize_param, Image.LANCZOS)
    elif isinstance(resize_param, (int, float)):
        # Resize proportionally
        width, height = img.size
        new_size = (int(width * resize_param), int(height * resize_param))
        return img.resize(new_size, Image.LANCZOS)
    else:
        raise ValueError("resize_param must be a scalar or a tuple.")

def create_gif(group_number, group_files, input_folder, output_folder, resize_param=None, padding_color=(255, 255, 255)):
    try:
        group_files = sorted(group_files)
        images = []
        max_width = 0
        max_height = 0

        # First pass: Resize images and determine maximum dimensions if padding is needed
        for file in group_files:
            image_path = os.path.join(input_folder, file)
            with Image.open(image_path) as img:
                if resize_param is not None:
                    img = resize_image(img, resize_param)

                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                images.append(img)

        # Check if all images are of the same size
        all_same_size = all(img.size == (max_width, max_height) for img in images)
        
        if not all_same_size:
            # If images are not of the same size, pad them
            padded_images = []
            for img in images:
                width, height = img.size
                left = (max_width - width) // 2
                top = (max_height - height) // 2
                right = max_width - width - left
                bottom = max_height - height - top

                # Create a new image with padding
                padded_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=padding_color)
                padded_images.append(padded_img)
            images = padded_images

        # Save the images as a GIF
        if images:
            gif_path = os.path.join(output_folder, f'Group_{group_number}.gif')
            images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=500)
            print(f"Saved GIF for Group {group_number} at {gif_path}")

    except Exception as e:
        print(f"Error processing group {group_number}: {e}")

def point_worm_gif(input_folder, output_folder, resize_param=None, padding_color=(0,0,0)):
    # Get and check file list
    tiff_files = [file for file in os.listdir(input_folder) if file.endswith(".tiff") or file.endswith(".tif")]

    
    # Regular expression to find the pattern *_Point(\d+)_*
    pattern = re.compile(r'_Point(\d+)')

    # Set to keep track of already processed group numbers
    processed_groups = set()

    # Create a list to store tasks for parallel processing
    tasks = []
    
    # Ensure the output folder exists, create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files to find and process each group
    for filename in tiff_files:
        match = pattern.search(filename)
        if match:
            group_number = int(match.group(1))

            if group_number not in processed_groups:
                processed_groups.add(group_number)
                group_files = [file for file in tiff_files if pattern.search(file) and int(pattern.search(file).group(1)) == group_number]
                tasks.append((group_number, group_files, input_folder, output_folder, resize_param, padding_color))

    # Process each group in parallel using joblib's Parallel
    Parallel(n_jobs=-3)(delayed(create_gif)(group_number, group_files, input_folder, output_folder, resize_param, padding_color) for group_number, group_files, input_folder, output_folder, resize_param, padding_color in tasks)




def plot_training_data(csv_file, output_folder):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    data = data[data.iloc[:, 0].notna() & (data.iloc[:, 0] != '')]
    print(data)

    # Identify columns ending with 'epoch', excluding the first 'epoch' column
    epoch_columns = [col for col in data.columns if col.endswith('epoch') and col != 'epoch']

    # Ensure the output folder exists, create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Set the path to save the plot
    output_file = os.path.join(output_folder, 'epoch_plot.png')

    # Plot each column ending with 'epoch' against the 'epoch' column on the same plot
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    for col in epoch_columns:
        subset = ['epoch', col]
        data_filtered = data.dropna(subset=subset)
        plt.plot(data_filtered['epoch'], data_filtered[col], label=col)
        print(data_filtered[col])

    # Manually set the y-axis limits between 0 and 1.5
    plt.ylim(0, 1)

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.title('Training Plot')
    plt.legend()

    # Save the plot to the specified folder
    plt.savefig(output_file)

    # Optionally, close the plot to free up memory
    plt.close()

    sys.stdout.write("Plot saved to {}".format(output_file))

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        sys.stderr.write("Usage: conversion.py <mode = 'worm', 'training'> <csv_file> <output_directory> opt: <col_prefix>")
        sys.exit(1)
        
    if len(sys.argv) == 5:
        col_prefix = True
    else:
        col_prefix = False

    if sys.argv[1] == 'worm':
        if col_prefix == True:
            plot_worm_data(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            plot_worm_data(sys.argv[2], sys.argv[3])
            
    if sys.argv[1] == 'training':
        plot_training_data(sys.argv[2], sys.argv[3])

    if sys.argv[1] == 'gif':
        point_worm_gif(sys.argv[2], sys.argv[3], 0.1)

    else:
        sys.stdout.write("Usage: conversion.py <mode = 'worm', 'training', 'gif'> <csv_file> <output_directory> opt: <col_prefix>") 
        sys.stderr.write("Mode has to be either 'worm', 'gif' or 'training'.")
        sys.exit(1)
        

    
