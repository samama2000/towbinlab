from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair."""
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    x = x / np.max(x)
    return x, y

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train StarDist Model on Provided Data")
    parser.add_argument('image_dir', type=str, help="Path to the folder containing training images")
    parser.add_argument('mask_dir', type=str, help="Path to the folder containing training masks")
    parser.add_argument('--model_name', default="stardist_emr_bigger_set", type=str, help="Name of the model to save")
    parser.add_argument('--model_dir', default="models", type=str, help="Directory where the model will be saved")
    return parser.parse_args()

def main():
    args = parse_args()

    # Get the directories from arguments
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    model_name = args.model_name
    model_dir = args.model_dir

    # Load images and masks
    X = sorted(glob(f'{image_dir}/*.tiff'))
    Y = sorted(glob(f'{mask_dir}/*.tiff'))
    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

    X = list(map(imread, X))
    Y = list(map(imread, Y))

    # Check if images are multi-channel and select only the last channel
    for i in range(len(X)):
        if X[i].ndim == 3 and X[i].shape[0] == 3:  # CXY format with 3 channels
            X[i] = X[i][1, :, :]  # Take only the last channel
            print(f"Selected last channel for image {i+1}.")
        elif X[i].ndim == 3 and X[i].shape[-1] == 3:  # YXC format with 3 channels
            X[i] = X[i][..., 1]  # Take only the last channel
            print(f"Selected last channel for image {i+1}.")

    n_channel = 1  # Now all images are single-channel after extracting the last channel

    axis_norm = (0, 1)   # normalize channels independently


    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    X = [x / np.max(x) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "Not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.25 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

    print('Number of images: %3d' % len(X))
    print('- Training:       %3d' % len(X_trn))
    print('- Validation:     %3d' % len(X_val))

    # Configuration parameters for training
    n_rays = 64
    use_gpu = gputools_available()
    print(f'Using {n_rays} rays and {"GPU" if use_gpu else "CPU"}.')

    grid = (4, 4)
    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        train_epochs=400,
        train_steps_per_epoch=100,
        train_patch_size=(512, 512),
    )

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(None, allow_growth=True)

    # Create model with user-provided model name and directory
    model = StarDist2D(conf, name=model_name, basedir=model_dir)

    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"Median object size:      {median_size}")
    print(f"Network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: Median object size larger than field of view of the neural network.")

    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)

if __name__ == '__main__':
    main()
