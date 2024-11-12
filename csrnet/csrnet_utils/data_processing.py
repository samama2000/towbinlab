import os
import logging
import numpy as np

import json
import base64
from io import BytesIO
from PIL import Image
from towbintools.foundation import image_handling
import torch

from sklearn.metrics import f1_score

from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border
from skimage.morphology import remove_small_objects
from scipy.spatial import distance

from .image_processing import (
    pad_image
)

# Function to decode base64 image data from JSON
def load_image_from_json(image_data):
    """
    Load the image data stored in base64 format inside the JSON file.
    Args:
    - image_data: base64-encoded image data from JSON.
    Returns:
    - image: Decoded image in numpy array format.
    """
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    return image


def save_results_as_json(image_path, centroids, output_path, channels=0):
    """
    Create a JSON file for LabelMe annotation from the results of your egg counting model.
    
    Args:
    - image_path (str): The file path to the input image.
    - centroids (list): A list of centroid coordinates [(x1, y1), (x2, y2), ...].
    - output_json_path (str): The file path where the JSON will be saved.
    - transform_dic (dict): The transformation dictionary with normalization parameters.

    
    Returns:
    - None. Writes the JSON file to the specified path.
    """
    
    # Load image and crop to right size

    image = image_handling.read_tiff_file(image_path, channels_to_keep=channels).squeeze()

    logging.info(f"image: {image}, {image.shape}")

    # Convert the centroids to Python-native types to avoid serialization issues
    centroids = [[int(c[1]), int(c[0])] for c in centroids]
    shapes = [{"label": "egg", "points": [list(c)]} for c in centroids]

    # Convert to base64
    pil_image = Image.fromarray(image.astype('uint8'))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_json_path = os.path.join(output_path, f"{image_basename}.json")

    # Create LabelMe compatible JSON structure
    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": "",
        "imageData": image_base64,
        "imageHeight": target_height,
        "imageWidth": target_width
    }

    # Write to the output JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)

    logging.info(f"JSON file saved to {output_json_path}")


def count_eggs_with_watershed(density_map, threshold=0.5, min_distance=20, min_size=20):
    density_map = density_map.detach().cpu().numpy() if isinstance(density_map, torch.Tensor) else density_map
    binary_map = density_map > threshold
    binary_map = clear_border(binary_map)
    
    # Remove small objects to reduce noise using built-in function
    binary_map_cleaned = remove_small_objects(binary_map, min_size=min_size)
    
    # Detect local maxima in the enhanced density map
    local_max_coords = peak_local_max(
        density_map, 
        min_distance=min_distance, 
        labels=binary_map_cleaned, 
        exclude_border=True
    )

    # Create an empty markers array with the same shape as binary_map_cleaned
    markers = np.zeros_like(binary_map_cleaned, dtype=int)
    
    # Place unique marker values at the local maxima positions
    for idx, coord in enumerate(local_max_coords):
        y, x = coord  # Coordinates of the local maxima
        markers[y, x] = idx + 1  # Unique marker value (starts from 1)

    # Extract centroids
    centroids = [(y, x) for y, x in local_max_coords]

    # Apply watershed segmentation
    watershed_labels = watershed(-density_map, markers, mask=binary_map_cleaned)
    egg_count = np.max(watershed_labels)
    logging.info(f"Number of watershed labels = Total eggs detected: {egg_count}")
     
    return egg_count, watershed_labels, centroids


def calculate_f1_score(output, target, threshold=0.50, min_size=20, min_distance=20, distance_threshold=25):
    batch_size = output.shape[0]
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(batch_size):
        # Extract the i-th image/tile from the batch
        output_single = output[i].squeeze(0)  # Remove channel dimension
        target_single = target[i].squeeze(0)  # Remove channel dimension

        # Perform watershed egg counting on both output and target
        output_count, output_watershed, output_centroids = count_eggs_with_watershed(output_single, threshold, min_distance, min_size)
        target_count, target_watershed, target_centroids = count_eggs_with_watershed(target_single, threshold, min_distance, min_size)

        logging.info(f"Image {i} - Output Count: {output_count}, Target Count: {target_count}")

        precision, recall, f1_score = compute_detection_metrics(output_centroids, target_centroids, distance_threshold)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    return precision_scores, recall_scores, f1_scores



def match_centroids(predicted_centroids, ground_truth_centroids, distance_threshold=25):
    """
    Match predicted centroids with ground truth centroids using a distance threshold.

    Args:
    - predicted_centroids: List of (x, y) coordinates of predicted egg centers.
    - ground_truth_centroids: List of (x, y) coordinates of ground truth egg centers.
    - distance_threshold: Maximum allowed distance to consider a match.

    Returns:
    - true_positive: Number of correctly matched predictions (TP).
    - false_positive: Number of false positives (FP).
    - false_negative: Number of false negatives (FN).
    """
    # Handle cases where there are no predicted or ground truth centroids
    if len(predicted_centroids) == 0:
        return 0, 0, len(ground_truth_centroids)  # No predictions, all ground truths are false negatives

    if len(ground_truth_centroids) == 0:
        return 0, len(predicted_centroids), 0  # No ground truths, all predictions are false positives

        
    # Track matched predictions
    matched_predictions = set()
    true_positive = 0

    for gt_idx, gt_centroid in enumerate(ground_truth_centroids):
        # Calculate the distances between the current ground truth and all predicted centroids
        distances = np.array([distance.euclidean(gt_centroid, pred) for pred in predicted_centroids])

        # Find the closest prediction
        min_dist = np.min(distances)
        closest_pred_idx = np.argmin(distances)

        # Check if the closest prediction is within the threshold
        if min_dist < distance_threshold:
            true_positive += 1
            matched_predictions.add(closest_pred_idx)

    false_positive = len(predicted_centroids) - len(matched_predictions)
    false_negative = len(ground_truth_centroids) - true_positive

    return true_positive, false_positive, false_negative


def compute_detection_metrics(predicted_centroids, ground_truth_centroids, distance_threshold=10):
    """
    Compute Precision, Recall, and F1 Score based on matched centroids.

    Args:
    - predicted_centroids: List of (x, y) coordinates of predicted egg centers.
    - ground_truth_centroids: List of (x, y) coordinates of ground truth egg centers.
    - distance_threshold: Maximum allowed distance to consider a match (in pixels).

    Returns:
    - precision: Precision of the predictions.
    - recall: Recall of the predictions.
    - f1_score: F1 score of the predictions.
    """
    tp, fp, fn = match_centroids(predicted_centroids, ground_truth_centroids, distance_threshold)

    # Calculate Precision, Recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

