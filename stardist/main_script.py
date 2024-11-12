import os
import sys
import time
import traceback
from pathlib import Path
from typing import Protocol

import cv2
import joblib
import numpy as np
import pandas as pd
import tifffile
from csbdeep.utils import normalize
from joblib import Parallel, delayed
from skimage import measure
from stardist.models import StarDist2D, StarDist3D
from tqdm import tqdm
from xgboost import XGBClassifier


class ImageFeatureClassifier(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def extract_image_features(self, image: np.ndarray) -> pd.DataFrame: ...


# stardist does not offer a 'list' of pretrained models. calling StarDist2D.from_pretrained() just prints available models to console
# so we define a list of models here to check against
DEFAULT_MODELS_2D = {
    "2D_versatile_fluo",
    "2D_versatile_he",
    "2D_paper_dsb2018",
}
# no pretrained 3D models are available in stardist at the moment. They have expressed the desire to add them in the future
# https://stardist.net/faq/#is-there-a-pretrained-model-for-3d-or-do-you-plan-to-release-one
DEFAULT_MODELS_3D = set()


def parallel_load_files(files, channel, threads=1):
    """
    returns a generator that loads images from files in parallel using multithreading

    generator is used to avoid loading all files into memory at once - they will be loaded on demand
    joblib preloads a few files ahead of time, so this is faster than loading them one by one
    """
    # since this is io, threading is sufficient and has lower overhead
    # return as generator to avoid loading all files into memory at once - they will be loaded on demand,
    # hopefully as fast as stardist can handle. Nonetheless, can still be rate limitting if cluster is slow
    image_generator = Parallel(
        n_jobs=threads, backend="threading", return_as="generator"
    )(delayed(tifffile.imread)(file, key=channel) for file in files)
    return image_generator


def square_crop(image, crop_axes=(-2, -1), return_slices: bool = False):
    """
    Crop a rectangular image to a square one along specified axes
    e.g. if image is 3x100x200, and crop_axes=(-2, -1), the output will be 3x100x100
    the resultant square is centered in the original image by taking the same amount of pixels from each side

    return_slices: if True, also returns the slices used to crop the image. These can be used to index into original image
    """
    if len(crop_axes) < 2:
        raise ValueError(
            "square_crop needs at least two axes to crop from. The smallest axis size determines the side length of the resultant image"
        )

    crop_axes = np.array(crop_axes)
    # convert negative axes to positive
    crop_axes = crop_axes % image.ndim

    crop_ax_sizes = [image.shape[ax] for ax in crop_axes]
    min_ax_size = min(crop_ax_sizes)

    # by default, keep everything from each axis
    crop_slices = [np.s_[:]] * image.ndim
    for ax in crop_axes:
        ax_size = image.shape[ax]
        size_delta = ax_size - min_ax_size
        crop_amount = size_delta // 2
        crop_from = crop_amount
        crom_to = image.shape[ax] - (size_delta - crop_amount)
        # overwrite the default slice with the cropped slice
        crop_slices[ax] = np.s_[crop_from:crom_to]

    crop_slices = tuple(crop_slices)
    cropped_image = image[crop_slices]
    if return_slices:
        return cropped_image, crop_slices
    else:
        return cropped_image


def preprocess_raw_image(image):
    # keep preprocessing minimal, stardist is by-and-large robust to _natural_ intensity variations,
    # and excessive preprocessing can lead to segmentation artifacts
    image = cv2.createCLAHE(clipLimit=6, tileGridSize=(8, 8)).apply(image)
    image = normalize(image)
    return image


# TODO: add logic to optionally square crop images


def stardist_seg(stardist_model: StarDist2D | StarDist3D, image, scale: float):
    image = preprocess_raw_image(image)
    # details unused. it contains data needed to e.g. draw/visualise polygons. see stardist examples on github if needed
    seg, details = stardist_model.predict_instances(image, scale=scale)
    return seg


def seg_analysis(properties, seg, image, spacing=(1, 1)):
    props_df = measure.regionprops_table(
        seg,
        intensity_image=image,
        properties=properties,
        spacing=spacing,
    )
    props_df = pd.DataFrame(props_df)
    return props_df

def filter_size_shape(props_df, min_size, max_size, min_solidity=0.8, max_eccentricity=0.95):
    """
    Filters out objects based on size and topology (e.g., solidity, eccentricity).
    
    Inputs:
        props_df (pd.DataFrame): Dataframe containing properties of segmented objects.
        min_size (int): Minimum area size to keep the object.
        max_size (int): Maximum area size to keep the object.
        min_solidity (float): Minimum solidity threshold to keep the object.
        max_eccentricity (float): Maximum eccentricity threshold to keep the object.
        
    Returns:
        pd.DataFrame: Filtered properties dataframe with objects meeting size and shape criteria.
    """
    filtered_df = props_df[
        (props_df['area'] >= min_size) & 
        (props_df['area'] <= max_size) & 
        (props_df['solidity'] >= min_solidity) & 
        (props_df['eccentricity'] <= max_eccentricity)
    ]
    return filtered_df

def stardist_seg_task(
    task_df,
    channel,
    stardist_model,
    xgboost_model,
    property_features,
    label_encoder,
    scale,
    body_mask_dir=None,
    crop=False,
    spacing=(1, 1),
    min_size=1000,  # Set a minimum size threshold for valid objects
    max_size=10000,  # Set a maximum size threshold for valid objects
    min_solidity=0.90,  # Minimum solidity threshold (optional topology filter)
    max_eccentricity=0.85  # Maximum eccentricity threshold (optional topology filter)
):
    input_files = task_df["input_file"].values
    output_files = task_df["output_file"].values

    input_images = parallel_load_files(input_files, channel=channel, threads=4)

    # Dictionary to map body mask files by their basename
    body_masks = {}
    if body_mask_dir:
        body_mask_files = os.listdir(body_mask_dir)
        for mask_file in body_mask_files:
            body_masks[os.path.basename(mask_file)] = os.path.join(body_mask_dir, mask_file)

    all_dfs = []
    for i, (raw_img, output_file) in tqdm(
        enumerate(zip(input_images, output_files)), total=len(input_files)
    ):
        if crop:
            image, crop_slices = square_crop(
                raw_img, crop_axes=(-2, -1), return_slices=True
            )
        else:
            image = raw_img
            crop_slices = tuple([np.s_[:]] * image.ndim)

        try:
            instance_seg = stardist_seg(stardist_model, image, scale)

            # Apply body mask if available for this image
            input_file_name = os.path.basename(input_files[i])
            body_mask = None
            if body_mask_dir and input_file_name in body_masks:
                body_mask = tifffile.imread(body_masks[input_file_name])
                instance_seg[body_mask > 0] = 0  # Mask out worm body areas
            
            # Extract region properties for filtering
            props_df = seg_analysis(property_features, instance_seg, image, spacing)

            # Apply filtering based on size and topology

            props_df = filter_size_shape(
                props_df, min_size, max_size, min_solidity, max_eccentricity
            )

            # Now update the segmentation to remove filtered objects
            filtered_seg = np.zeros_like(instance_seg)
            for _, region in props_df.iterrows():
                label = region['label']
                filtered_seg[instance_seg == label] = label

            # Save the filtered segmentation
            uncropped_seg = np.zeros(raw_img.shape, dtype=filtered_seg.dtype)
            uncropped_seg[crop_slices] = filtered_seg
            tifffile.imwrite(output_file, uncropped_seg)
        except Exception as e:
            print(
                f"Error doing stardist instance segmentation for {input_files[i]}: {e}"
            )
            traceback.print_exc()
            continue

        if xgboost_model is not None:
            try:
                features = props_df.drop(columns="label")
                props_df["encoded_class"] = xgboost_model.predict(features.values)
                props_df["class"] = label_encoder.inverse_transform(
                    props_df["encoded_class"].values
                )

                props_df["Time"] = task_df["Time"].values[i]
                props_df["Point"] = task_df["Point"].values[i]

                all_dfs.append(props_df)
            except Exception as e:
                print(
                    f"Error classifying instance segmentation for {input_files[i]}: {e}"
                )
                traceback.print_exc()
                continue

    if len(all_dfs) == 0:
        return None
    df = pd.concat(all_dfs, axis=0)
    return df


def define_task(
    filemap: pd.DataFrame,
    input_file_col: str,
    output_dir: str,
    point: int,
    time_range: tuple[int, int],
):
    task_df = filemap[["Time", "Point", input_file_col]].copy()
    task_df = task_df.rename(columns={input_file_col: "input_file"})

    task_df = task_df.dropna(subset=["input_file"])
    task_df = task_df.astype({"Time": int, "Point": int, "input_file": str})

    start_time, end_time = time_range
    if end_time == -1:
        end_time = task_df["Time"].max()
    task_df = task_df[
        (task_df["Time"].between(start_time, end_time, inclusive="both"))
        & (task_df["Point"] == point)
    ]

    task_df["output_file"] = (
        task_df["input_file"]
        .transform(lambda file: Path(output_dir) / Path(file).name)
        .astype(str)
    )

    print(f"Processing {len(task_df)} images for point {point}...")
    return task_df


def load_stardist_model(model_name: str) -> StarDist2D | StarDist3D:
    # stardist automatically loads the model onto gpu
    # if no gpu is present, cpus will be used
    if model_name in DEFAULT_MODELS_2D:
        model = StarDist2D.from_pretrained(model_name)
    elif model_name in DEFAULT_MODELS_3D:
        model = StarDist3D.from_pretrained(model_name)
    else:
        # if wish to add custom models, add a 'load_model' function here
        # it was not obvious to me how to leverage stardist to load a custom model, even when following their example notebooks

        # model_path = Path(model_name)
        # assert model_path.exists(), f"Model {model_name} not found"
        # model = ...
        raise ValueError(
            f"Model {model_name} not found in default stardist models. Custom models not yet supported."
        )
    if model is None:
        raise ValueError(
            f"Loading model {model_name} went wrong: returned model is None"
        )

    return model


def load_xgboost_model(model_path: str) -> tuple[XGBClassifier, list[str]]:
    model_dict = joblib.load(model_path)

    xgboost_model, feature_properties, label_encoder = (
        model_dict["model"],
        model_dict["property_features"],
        model_dict["label_encoder"],
    )

    return xgboost_model, feature_properties, label_encoder


def parse_args(args=sys.argv[1:]):
    import argparse

    # ~/.local/bin/micromamba run -n stardist_env python3 main_script.py --filemap $FILEMAP_CSV --out_dir $STARDIST_SEG_OUT_DIR --scale $SCALE --crop $CROP_TO_SQUARE --t_start $TIME_START --t_end $TIME_END --channel $CHANNEL --column $COLUMN --xgboost $XGBOOST_MODEL --stardist $STARDIST_MODEL$"

    parser = argparse.ArgumentParser(description="Stardist Segmentation Script")
    parser.add_argument("--filemap", type=str, help="Path to the filemap CSV file")
    parser.add_argument(
        "--out_dir", type=str, help="Output directory for segmentation results"
    )
    parser.add_argument(
        "--scale", type=float, help="Scale parameter for Stardist segmentation"
    )
    parser.add_argument(
        "--crop",
        type=lambda crop: crop.lower() == "true",
        help="Flag to indicate whether to crop images to square",
    )
    parser.add_argument("--t_start", type=int, help="Start time for segmentation")
    parser.add_argument("--t_end", type=int, help="End time for segmentation")
    parser.add_argument("--channel", type=int, help="Channel for segmentation")
    parser.add_argument("--column", type=str, help="Column for segmentation")
    parser.add_argument("--xgboost", type=str, help="Path to the XGBoost model")
    parser.add_argument("--stardist", type=str, help="Name of the Stardist model")
    parser.add_argument(
        "--spacing", type=lambda s: tuple(map(int, s.split(","))), default=(1, 1)
    )
    parser.add_argument(
        "--body_mask_dir", type=str, help="Directory containing body mask image files"  # Update argument to directory
        )

    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    start_time = time.time()
    try:
        SLURM_ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID")
    except KeyError:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID not set - this script should be run as a job array"
        )

    args = parse_args(args)

    point = int(SLURM_ARRAY_TASK_ID)

    print(args)

    print(
        f"{time.asctime()} - Starting Stardist segmentation for point {SLURM_ARRAY_TASK_ID}"
    )
    stardist_model = load_stardist_model(args.stardist)
    xgboost_model, property_features, label_encoder = load_xgboost_model(args.xgboost)
    filemap = pd.read_csv(args.filemap)
    task_df = define_task(
        filemap=filemap,
        input_file_col=args.column,
        output_dir=args.out_dir,
        point=point,
        time_range=(args.t_start, args.t_end),
    )

    point_seg_features = stardist_seg_task(
        task_df=task_df,
        channel=args.channel,
        stardist_model=stardist_model,
        xgboost_model=xgboost_model,
        property_features=property_features,
        label_encoder=label_encoder,
        scale=args.scale,
        crop=args.crop,
        spacing=args.spacing,
        body_mask_dir=args.body_mask_dir,
    )

    if point_seg_features is not None:
        csv_dir = Path(args.filemap).parent / "stardist_seg_features"
        csv_dir.mkdir(exist_ok=True, parents=True)

        csv_path = csv_dir / f"point_{point}_features.csv"
        point_seg_features.to_csv(csv_path, index=False)

    end_time = time.time()
    print(f"{time.asctime()} - Finished Stardist segmentation for point {point}...")

    secs_duration = int(round((end_time - start_time)))
    print(f"Time taken: {secs_duration}s to process {len(task_df)} images")


if __name__ == "__main__":
    # args = "--filemap /mnt/towbin.data/shared/smarin/data/2024_02_07_lifespan_wBT318_20C/analysis/report/analysis_filemap.csv --out_dir /mnt/towbin.data/shared/smarin/data/2024_02_07_lifespan_wBT318_20C/analysis/stardist_seg --scale 0.5 --crop True --t_start 0 --t_end -1 --channel 1 --column raw --xgboost /mnt/towbin.data/shared/bgusev/stardist_egg_seg/xgboost_training/egg_classifier_models/egg_classifier_v1.joblib.pkl --stardist 2D_versatile_fluo".split()
    # os.environ["SLURM_ARRAY_TASK_ID"] = "35"
    # main(args)

    main()
