# logic
import os
import numpy as np
import math as mt
import torch


# Modified padding function using reflection padding
def pad_image(image, target_height, target_width):
    """
    Pad the image to a target size (height and width) using reflection padding.
    Args:
    - image: Input image (numpy array, HxW or HxWxC).
    - target_height: The desired height of the padded image.
    - target_width: The desired width of the padded image.
    Returns:
    - padded_image: Padded image.
    - padding_info: The amount of padding added to each side (top, bottom, left, right).
    """
    height, width = image.shape[:2]
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    padding_top = pad_height // 2
    padding_bottom = pad_height - padding_top
    padding_left = pad_width // 2
    padding_right = pad_width - padding_left

    padding_info = (padding_top, padding_bottom, padding_left, padding_right)

    if len(image.shape) == 3:  # For RGB or 3D arrays
        padded_image = np.pad(image, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='reflect')
    else:  # For grayscale or 2D arrays
        padded_image = np.pad(image, ((padding_top, padding_bottom), (padding_left, padding_right)), mode='reflect')

    return padded_image, padding_info

# Cropping function to ensure consistency in image size
def crop_image(image, target_height, target_width):
    """
    Crop the image to the target size (if necessary).
    Args:
    - image: Input image (numpy array).
    - target_height: Desired height of the image.
    - target_width: Desired width of the image.
    Returns:
    - Cropped image of size (target_height, target_width).
    """
    height, width = image.shape[:2]

    if height > target_height:
        start_y = (height - target_height) // 2
        image = image[start_y:start_y + target_height, :]

    if width > target_width:
        start_x = (width - target_width) // 2
        image = image[:, start_x:start_x + target_width]

    return image

# Tiling function for breaking images into tiles
def tile_image(image, tile_size=500, overlap=0):
    """
    Break the image into overlapping tiles.
    Args:
    - image: Input image (numpy array).
    - tile_size: Desired tile size.
    - overlap: Amount of overlap between tiles.
    Returns:
    - tiles: List of image tiles.
    - coords: Coordinates of the top-left corner of each tile.
    """
    tiles = []
    coords = []
    height, width = image.shape[:2]

    for y in range(0, height - tile_size + 1, tile_size - overlap):
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            coords.append((y, x))

    return tiles, coords

def stitch_tiles(tiles, coords, original_shape, changed_shape, device, tile_size, overlap):
    """
    Stitch tiles back together considering overlaps.

    Args:
    - tiles: List of predicted tiles (from the model).
    - coords: List of (y, x) coordinates for the top-left corner of each tile.
    - original_shape: The original shape of the full image before tiling.
    - tile_size: Size of each tile.
    - overlap: Amount of overlap between the tiles.
    
    Returns:
    - stitched_image: The full stitched image.
    """
    # Ensure all tiles are converted to tensors (if they are not already)
    tiles = [torch.tensor(tile) if not isinstance(tile, torch.Tensor) else tile for tile in tiles]
    
    # Initialize stitched image and a count matrix for averaging overlaps
    stitched_image = torch.zeros(changed_shape, dtype=tiles[0].dtype, device=device)
    counts = torch.zeros(changed_shape, dtype=torch.float32, device=device)

    for tile, (y, x) in zip(tiles, coords):
        tile = tile.squeeze(0)
        tile = tile.to(device)

        y_end = y + tile_size
        x_end = x + tile_size

        # Handle overlap by determining the actual size to copy from the tile
        tile_y_end = tile_size if y_end <= stitched_image.shape[0] else stitched_image.shape[0] - y
        tile_x_end = tile_size if x_end <= stitched_image.shape[1] else stitched_image.shape[1] - x

        # Add the tile to the stitched image and update counts
        stitched_image[y:y_end, x:x_end] += tile[0, :tile_y_end, :tile_x_end]  # Using [0, ...] to remove channel dimension
        counts[y:y_end, x:x_end] += 1

    # Avoid division by zero and average the stitched image
    stitched_image /= torch.maximum(counts, torch.ones_like(counts))
    
    cropped_stitched_image = crop_image(stitched_image, original_shape[0], original_shape[1])

    return cropped_stitched_image

