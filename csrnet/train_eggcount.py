import os
import json

import numpy as np
import math as mt
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


from csrnet_utils.csrnet import CSRNet

from csrnet_utils.utils import (
    load_config,
    setup_logging,
    save_best_model
)

from csrnet_utils.image_processing import (
    pad_image, 
    crop_image, 
    tile_image, 
    stitch_tiles, 
)

from csrnet_utils.data_processing import (
    load_image_from_json,
    save_results_as_json,
    count_eggs_with_watershed,
    calculate_f1_score,
)

from csrnet_utils.plotting import (
    save_image_with_overlay,
    save_tile_image
)

from towbintools.deep_learning.utils.augmentation import (
    get_training_augmentation,
    get_prediction_augmentation,
)

from scipy.ndimage import gaussian_filter

import logging




# Function to generate a density map from points
def generate_density_map(image_shape, points, sigma=10):
    """
    Generate a density map for the image based on annotated points.
    Args:
    - image_shape: Tuple (height, width) of the image.
    - points: List of (x, y) coordinates for the annotated egg centers.
    - sigma: Standard deviation for the Gaussian kernel.
    Returns:
    - density_map: A density map with Gaussian kernels at each point location.
    """
    # Generate density map
    density_map = np.zeros(image_shape, dtype=np.float32)
        
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x >= image_shape[1] or y >= image_shape[0]:  # Ensure points are within bounds
            continue
        density_map[y, x] = 1

    # Apply Gaussian filter to create density map
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')

    # Normalise density map
    max_value = np.max(density_map)
    normalised_density_map = density_map / max_value

    return normalised_density_map


def validate_csrnet(model, val_dataloader, criterion, device, threshold, min_size, min_distance, match_distance):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for i, (images, density_maps, _, _) in enumerate(val_dataloader):
            images = images.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images)
            
            loss = criterion(outputs, density_maps)
            precision, recall, f1 = calculate_f1_score(outputs, density_maps, threshold, min_size, min_distance, match_distance)

            total_loss += loss.item()
            total_precision += precision[0]
            total_recall += recall[0]
            total_f1 += f1[0]
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    return avg_loss, avg_precision, avg_recall, avg_f1


# Custom Dataset class for loading images and density maps with cropping and padding
class EggCountingDataset(Dataset):
    def __init__(self, json_dir, tile_size=None, overlap=0, label='egg'):
        """
        Args:
        - json_dir: Path to the directory containing LabelMe JSON files.
        - tile_size: Desired tile size for tiling (optional). If None, use the original image size.
        - overlap: Overlap between tiles if tiling is used.
        - transform: Transformations to be applied to the images (e.g., normalization).
        """
        self.json_dir = json_dir
        self.json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
        self.tile_size = tile_size
        self.overlap = overlap
        self.label = label
        
        # Preprocess tiles or full images depending on the tile_size
        self.tiles = []
        self.tile_coords = []
        self.density_map_tiles = []
        self.image_ids = []
        

        # Get the original size from the first image in the dataset
        if len(self.json_files) > 0:
            with open(self.json_files[0], 'r') as f:
                data = json.load(f)
                self.original_height = data['imageHeight']
                self.original_width = data['imageWidth']
                
        
        # Calculate the target size based on tile size or original image size
        if self.tile_size:
            self.target_height = int(mt.ceil((self.original_height - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)
            self.target_width = int(mt.ceil((self.original_width - self.overlap) / (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.overlap)
        else:
            self.target_height = self.original_height
            self.target_width = self.original_width
            

        for image_idx, json_file in enumerate(self.json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)

            image_data = data['imageData']
            image = load_image_from_json(image_data)
            points = [shape['points'][0] for shape in data['shapes'] if shape['label'] == self.label]
            image_shape = (data['imageHeight'], data['imageWidth'])
            density_map = generate_density_map(image_shape, points)

            if self.tile_size:
                # If tiling is enabled
                padded_image, _ = pad_image(image, self.target_height, self.target_width)
                padded_density_map, _ = pad_image(density_map, self.target_height, self.target_width)

                image_tiles, image_coords = tile_image(padded_image, self.tile_size, self.overlap)
                density_map_tiles, _ = tile_image(padded_density_map, self.tile_size, self.overlap)

                self.tiles.extend(image_tiles)
                self.tile_coords.extend(image_coords)
                self.density_map_tiles.extend(density_map_tiles)
                self.image_ids.extend([image_idx] * len(image_tiles))

            else:
                # If tiling is disabled (full images)
                cropped_image = crop_image(image, self.target_height, self.target_width)
                cropped_density_map = crop_image(density_map, self.target_height, self.target_width)
                
                self.tiles.append(cropped_image)
                self.tile_coords.append(None)  # No coords for full images
                self.density_map_tiles.append(cropped_density_map)
                self.image_ids.append(image_idx)

    
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image_tile = self.tiles[idx]
        density_map_tile = self.density_map_tiles[idx]
        image_id = self.image_ids[idx]
        coord = self.tile_coords[idx]

        return image_tile, density_map_tile, image_id, coord
            
        
# Custom Dataset wrapper to apply specific transforms for each split
class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, density_map, image_id, coord = self.base_dataset[idx]

        # Apply the transform if available
        if self.transform:
            transformed = self.transform(image=image, mask=density_map)
            image = transformed['image']
            density_map = transformed['mask']
        
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        density_map_tensor = torch.FloatTensor(density_map).unsqueeze(0)

        # Return the transformed image along with other data
        return (image_tensor, density_map_tensor, image_id, coord)


def train_csrnet_with_tiling(config):
    # Load settings from config

    # File settings
    json_dir = config['json_directory']
    save_dir = config['save_dir']
    log_file = config.get('log_file', 'training.log')
    num_best_models = config.get('num_best_models', 4)  # Number of best models to save
    label = config['label']

    # GPU settings
    use_gpu = config.get('use_gpu', False)
    gpu_device = config.get('gpu_device', None)

    # Model settings
    pretrained = config['pretrained']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    transform_dic = config.get('transform', None)
    tile_size = config.get('tile_size', None)  # Optional tile size
    overlap = config.get('overlap', 0)

    # Count settings
    threshold = config.get('threshold', 0.5)
    min_size = config.get('min_size', 20)
    min_distance = config.get('min_distance', 25)
    match_distance = config.get('match_distance', 25)

    # Visualisation
    tile_dir = config.get('tile_dir', None)
    

    # Create necessary paths
    os.makedirs(save_dir, exist_ok = True)
    if tile_dir:
        os.makedirs(tile_dir, exist_ok = True)

    
    # Setup Transformation
    if transform_dic is not None:
        transform_type = transform_dic["type"]
        transform_parameters = {k: v for k, v in transform_dic.items() if k != "type"}
        training_transform = get_training_augmentation(transform_type, **transform_parameters)
        validation_transform = get_prediction_augmentation(transform_type, **transform_parameters)
    else:
        training_transform = None
        validation_transform = None



    # Set up logging
    log_file_path = os.path.join(save_dir, log_file)
    setup_logging(log_file_path)
    logging.info(f"Starting training with config: {config}")

    # Initialize dataset with automatic size calculation
    full_dataset = EggCountingDataset(json_dir, tile_size=tile_size, overlap=overlap, label=label)

    # Split the dataset into training (75%) and validation (25%) sets
    train_size = int(0.75 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Wrap the datasets with the custom transform class
    train_dataset = TransformDataset(train_dataset, transform=training_transform)
    val_dataset = TransformDataset(val_dataset, transform=validation_transform)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)  # Batch size 1 for validation

    model = CSRNet(transform_dic = transform_dic, pretrained = pretrained).cuda() if use_gpu else CSRNet(transform_dic = transform_dic, pretrained = pretrained)

    # Use DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if use_gpu:
        device = torch.device(f"cuda:{gpu_device[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model.to(device)

    # Track the best models
    saved_models = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i, (images, density_maps, _, _) in enumerate(train_dataloader):
            images = images.to(device)
            density_maps = density_maps.to(device)

            # Get model predictions
            outputs = model(images)
            # Compute loss
            loss = criterion(outputs, density_maps)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

            # Calculate F1 score
            precisions, recalls, f1s = calculate_f1_score(outputs, density_maps, threshold, min_size, min_distance, match_distance)
            precision_scores.extend(precisions)
            recall_scores.extend(recalls)
            f1_scores.extend(f1s)

            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_dataloader)}, Loss: {loss.item()}, F1 Score: {f1s}")

             # Save random tiles for inspection (optional, save every 5 batches)
            if tile_dir is not None and i % 20 == 0:
                random_idx = random.randint(0, images.shape[0] - 1)
                save_tile_image(images[random_idx], density_maps[random_idx], outputs[random_idx], epoch, i, tile_dir, threshold)
        
        # Free GPU memory after each epoch
        torch.cuda.empty_cache()

        # Average F1 score for the epoch
        avg_f1 = np.mean(f1_scores)
        logging.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {running_loss / len(train_dataloader)}, Average F1 Score: {avg_f1}")
        
        # Validation loop with stitching
        val_loss, val_precision, val_recall, val_f1 = validate_csrnet(model, val_dataloader, criterion, device, threshold, min_size, min_distance, match_distance)

        logging.info(f"Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}")

        # Save the model if it's among the best N
        save_best_model(model, epoch, val_f1, saved_models, num_best_models, save_dir)

    logging.info("Training complete.")







# Entry point to load config and train the model
if __name__ == '__main__':
    import argparse

    # Set up argument parser to receive config file
    parser = argparse.ArgumentParser(description='Train CSRNet with optional Tiling using YAML Config')
    parser.add_argument('config', type=str, help='Path to the config YAML')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train CSRNet with tiling based on the config
    train_csrnet_with_tiling(config)

