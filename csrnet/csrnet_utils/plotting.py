import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from towbintools.foundation import image_handling
from .data_processing import count_eggs_with_watershed
from .image_processing import (
    crop_image,
    pad_image,
    tile_image,
    stitch_tiles
)


def save_tile_image(image, density_map, output_map, epoch, batch, save_dir, threshold=0.5):
    # Convert tensors to numpy arrays and remove extra dimensions
    image = image.detach().cpu().squeeze().numpy()
    target = density_map.detach().cpu().squeeze().numpy()
    output = output_map.detach().cpu().squeeze().numpy()

    output_count, watershed_output, _ = count_eggs_with_watershed(output, threshold)  # Keep on GPU if possible
    target_count, watershed_target, _ = count_eggs_with_watershed(target, threshold)   # Keep on GPU if possible


    # Create a plot of the image, ground truth, and prediction
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(target, cmap='jet')
    ax[1].set_title(f"Ground Truth (Eggs: {target_count})")
    ax[1].axis('off')

    ax[2].imshow(output, cmap='jet')
    ax[2].set_title(f"Prediction (Eggs: {output_count})")
    ax[2].axis('off')

    # Save the plot with epoch and batch info
    save_path = f"{save_dir}/epoch_{epoch}_batch_{batch}.png"
    fig.suptitle(f"Epoch {epoch} Batch {batch} - Pred: {output_count}, True: {target_count}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    
def save_image_with_overlay(watershed_labels, image_path, channels, density_map, egg_count, output_dir, clip_limit = 5, augment_contrast = False):
    os.makedirs(output_dir, exist_ok=True)

    image = image_handling.read_tiff_file(image_path, channels_to_keep=channels).squeeze()
    if augment_contrast:
        image = image_handling.augment_contrast(image, clip_limit=clip_limit)

    if isinstance(density_map, torch.Tensor):
        density_map = density_map.detach().cpu().numpy()
    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image, cmap='gray')  # Show the grayscale original image
    
    # Overlay the labeled image as red contours
    ax.contour(watershed_labels, levels=np.unique(watershed_labels), colors='red', linewidths=0.5)  # Red contours

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Set the title and axis
    ax.set_title(f"Image {basename} - Eggs: {egg_count}")
    ax.axis('off')

    # Save the figure
    save_path = os.path.join(output_dir, f"{basename}_eggs_{egg_count}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)