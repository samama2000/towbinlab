### Imports ###
# logic
import os
import numpy as np
import math as mt
import csv
import re

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# image processing
import tifffile as tf

from towbintools.foundation import image_handling

from towbintools.deep_learning.utils.augmentation import (
    get_training_augmentation,
    get_prediction_augmentation,
)

from csrnet_utils.csrnet import CSRNet

from csrnet_utils.utils import (
    load_config,
    setup_logging,
    load_model
)

from csrnet_utils.image_processing import (
    pad_image, 
    crop_image, 
    tile_image, 
    stitch_tiles
)

from csrnet_utils.data_processing import (
    save_results_as_json,
    count_eggs_with_watershed
)

from csrnet_utils.plotting import (
    save_image_with_overlay
)




# bonus
import logging
import concurrent.futures



### SETUP ### -------------------------------------------------------------------------------------------

def save_egg_counts_to_csv(egg_counts, output_dir):
    output_csv_path = os.path.join(output_dir, 'results.csv')

    # Extract and sort the egg counts by Time and Point
    sorted_egg_counts = sorted(
        egg_counts.items(),
        key=lambda x: (
            int(re.search(r'Time(\d+)_Point(\d+)', x[0]).group(1)),  # Sort by Time
            int(re.search(r'Time(\d+)_Point(\d+)', x[0]).group(2))   # Then sort by Point
        )
    )

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Time', 'Point', 'egg_count'])  # Write the header

        for image_id, egg_count in sorted_egg_counts.items():
            # Extract Time and Point from the image_id using regex
            match = re.search(r'Time(\d+)_Point(\d+)', image_id)
            if match:
                time_value = int(match.group(1))
                point_value = int(match.group(2))
                # Write the row to the CSV
                writer.writerow([time_value, point_value, egg_count])

    logging.info(f"Egg counts saved to {output_csv_path}")

### DATASET ### -------------------------------------------------------------------------------------------

class EggCountingDataset(Dataset):
    def __init__(self, image_dir, channels=[0], tile_size=None, overlap=0, augment_contrast=False, clip_limit=5, num_workers=16, transform = None):
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if (f.endswith('.tiff') or f.endswith('.tif'))
            and (match := re.search(r'Point(\d{4})', f))  # Match 'Point' followed by 4 digits
            and int(match.group(1)) < 97  # Ensure the Point number is 103 or less
        ]
        logging.info(f"{self.image_files}")
        self.channels = channels
        self.augment_contrast = augment_contrast
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.overlap = overlap
        self.num_workers = num_workers
        self.transform = transform
        

        # Get the original size from the first image in the dataset if necessary
        if not self.tile_size and len(self.image_files) > 0:
            self.data_shape = tf.TiffFile(os.path.join(self.image_files[0], image_dir)).pages[0].shape[:2]

        self.image_tile_indices = {}
        self.setup_image_index_map()
        
        
    def setup_image_index_map(self):
        current_index = 0
        for image_file in self.image_files:
            image_id = str(image_file)
            with tf.TiffFile(image_file) as tif:
                height, width = tif.pages[0].shape
                
            if self.tile_size:
                target_height = int(mt.ceil((height - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)
                target_width = int(mt.ceil((width - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)
                num_tiles_x = ((target_width - self.tile_size) // (self.tile_size - self.overlap)) + 1
                num_tiles_y = ((target_height - self.tile_size) // (self.tile_size - self.overlap)) + 1
                num_tiles = num_tiles_x * num_tiles_y
            else:
                num_tiles = 1

            self.image_tile_indices[image_id] = (current_index, current_index + num_tiles)
            current_index += num_tiles



    def get_tile(self, image_idx, tile_idx):
        image = image_handling.read_tiff_file(image_idx, channels_to_keep=self.channels).squeeze()
        
        if self.augment_contrast:
            image = image_handling.augment_contrast(image, clip_limit=self.clip_limit)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']


        if self.tile_size:
            target_height = int(mt.ceil((image.shape[0] - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)
            target_width = int(mt.ceil((image.shape[1] - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)

            padded_image, _ = pad_image(image, target_height, target_width)
            image_tiles, image_coords = tile_image(padded_image, self.tile_size, self.overlap)

            return image_tiles[tile_idx], image_coords[tile_idx], image.shape, padded_image.shape, image_idx

        else:
            cropped_image = crop_image(image, self.data_shape[0], self.data_shape[1])

            return [cropped_image], [None], image.shape, cropped_image.shape, image_idx



    def __len__(self):
        last_image_id = list(self.image_tile_indices.keys())[-1]
        total_tiles = self.image_tile_indices[last_image_id][1]
        return total_tiles

    def __getitem__(self, idx):
        for image_idx, (start_idx, end_idx) in self.image_tile_indices.items():
            if start_idx <= idx < end_idx:
                # Determine the tile index relative to the image's tiles
                tile_idx = idx - start_idx
                image_tile, tile_coord, original_image_shape, changed_image_shape, image_id = self.get_tile(image_idx, tile_idx)
                
        image_tile_tensor = torch.FloatTensor(image_tile).unsqueeze(0)
        
        
        return image_tile_tensor, tile_coord, original_image_shape, changed_image_shape, image_id

### EVALUATION ### -------------------------------------------------------------------------------------------

def count_eggs(config):
    """
    Count eggs in each image by processing its tiles and stitching them back together.
    
    Args:
    - model: Trained CSRNet model for counting.
    - dataset: EggCountingDataset that returns individual tiles.
    - tile_size: Size of each tile.
    - overlap: Overlap between tiles.
    - device: Device for processing (CPU or CUDA).
    
    Returns:
    - egg_counts: Dictionary mapping image indices to predicted egg counts.
    """
    # Load settings from config
    image_dir = config['image_dir']
    output_dir = config['output_dir']
    channels = config.get('channels', [0])
    augment_contrast = config.get('augment_contrast', False)
    clip_limit = config.get('clip_limit', 5)
    threshold = config.get('threshold', 0.5)
    min_size = config.get('min_size', 20)
    min_distance = config.get('min_distance', 25)
    model_path = config['model_path']
    batch_size = config.get('batch_size', 4)
    tile_size = config.get('tile_size', None)  # Optional tile size
    overlap = config.get('overlap', 0) # Optional overlap
    log_file = config.get('log_file', 'eggcount.log')
    num_workers = config.get('num_workers', 18)
    create_json = config.get('create_json', False)
    save_visual = config.get('save_visual', False)
    
    use_gpu = config.get('use_gpu', False)
    gpu_device = config.get('gpu_device', None)

    os.makedirs(output_dir, exist_ok=True)
    
    if create_json:
        json_path = os.path.join(output_dir, "point_anno")
        os.makedirs(json_path, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(output_dir, log_file)
    
    setup_logging(log_file_path)
    logging.info(f"Starting training with config: {config}")

    model, device = load_model(
        model_path, 
        use_gpu=use_gpu, 
        gpu_device=gpu_device
        )
    
    # Check if the model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        transform_dic = model.module.transform_dic  # Access the underlying model
    else:
        transform_dic = model.transform_dic
    if transform_dic is None:
        transform = None
    else:
        transform_type = transform_dic["type"]
        transform_parameters = {k: v for k, v in transform_dic.items() if k != "type"}
        transform = get_prediction_augmentation(transform_type, **transform_parameters)


    
    model.eval()  # Set model to evaluation mode
    egg_counts = {}  # Dictionary to store egg counts for each image
    
    dataset = EggCountingDataset(
        image_dir, 
        channels=channels, 
        tile_size=tile_size, 
        overlap=overlap, 
        augment_contrast=augment_contrast, 
        num_workers=num_workers,
        transform=transform
        )
    

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
        )
    
    logging.info(f"Dataset processed.")

    # Variables to hold tiles for the current image
    current_image_idx = None
    stitched_output = None
    coord_list = []
    
    with torch.no_grad():  # No need to compute gradients during inference
        for batch, (image_tiles, tile_coords, original_image_shapes, changed_image_shapes, image_idx) in enumerate(dataloader):
            

            # Correct tile_coords
            y_coords, x_coords = tile_coords
            tile_coords = list(zip(y_coords.tolist(), x_coords.tolist()))

            # Correct image shapes
            original_height, original_width = original_image_shapes
            original_image_shapes = list(zip(original_height.tolist(), original_width.tolist()))
            
            changed_height, changed_width = changed_image_shapes
            changed_image_shapes = list(zip(changed_height.tolist(), changed_width.tolist()))


            image_tiles = image_tiles.to(device)

            # Process each tile individually
            for i in range(len(image_idx)):                   
                image_tile = image_tiles[i:i+1]  # Select the i-th tile
                tile_coord = tile_coords[i]
                original_image_shape = original_image_shapes[i]
                changed_image_shape = changed_image_shapes[i]
                image_id = str(image_idx[i])
                
            
                # Check if this is a new image
                if current_image_idx is None or image_id != current_image_idx:
                    # If there are tiles from a previous image, stitch and count eggs
                    if stitched_output is not None:
                        full_density_map = stitch_tiles(stitched_output, coord_list, current_original_shape, current_changed_shape, device, tile_size, overlap)

                        egg_count, watershed_labels, centroids = count_eggs_with_watershed(full_density_map, threshold = threshold, min_size = min_size, min_distance = min_distance)

                        if save_visual:
                            save_image_with_overlay(
                                watershed_labels, 
                                current_image_idx,
                                channels, 
                                full_density_map, 
                                egg_count, 
                                output_dir, 
                                clip_limit, 
                                augment_contrast
                            )


                        if create_json and egg_count > 0:
                            save_results_as_json(current_image_idx, centroids, json_path, channels)

                        egg_counts[current_image_idx] = egg_count


                        logging.info(f"Image {current_image_idx} - Predicted Eggs: {egg_count}")
                    
                    # Reset for the new image
                    current_image_idx = image_id
                    stitched_output = []
                    coord_list = []
                    current_original_shape = original_image_shape
                    current_changed_shape = changed_image_shape
            
                # Process the tile through the model
                output = model(image_tile)
                stitched_output.append(output.detach().cpu().numpy())
                coord_list.append(tile_coord)

                logging.info(f"Processing Tile {(batch * batch_size) + i}/{len(dataset)}")
            
        
        # Handle the final image after the loop
        if stitched_output is not None:
            full_density_map = stitch_tiles(stitched_output, coord_list, current_original_shape, current_changed_shape, device, tile_size, overlap)
            egg_count, watershed_labels, centroids = count_eggs_with_watershed(full_density_map)
            save_image_with_overlay(
                watershed_labels, 
                current_image_idx, 
                channels, 
                full_density_map, 
                egg_count, 
                output_dir, 
                clip_limit, 
                augment_contrast
            )

            if create_json and egg_count > 0:
                save_results_as_json(current_image_idx, centroids, json_path, channels)
                            
            egg_counts[current_image_idx] = egg_count

            logging.info(f"Final Image {current_image_idx} - Predicted Eggs: {egg_count}")

        logging.info(f"Processed all images. Saved image paths at: {output_dir}.")
    
    save_egg_counts_to_csv(egg_counts, output_dir)

    logging.info(f"Saved result csv in path: {output_dir}")

    return egg_counts

### MAIN ### -------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Egg Counting with CSRNet')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Count eggs
    egg_counts = count_eggs(config)

