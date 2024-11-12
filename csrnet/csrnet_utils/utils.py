# logic
import os
import torch
import torch.nn as nn

import logging
import yaml
import heapq

from .csrnet import CSRNet

# Function to read YAML config file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


# Set up the logging system
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
def load_model(model_path, use_gpu=False, gpu_device=None):
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cuda' if use_gpu else 'cpu')

    # Extract state_dict and transform_dic
    state_dict = checkpoint.get('model_state_dict', None)
    transform_dic = checkpoint.get('transform_dic', {'type': 'percentile', 'lo': 1, 'hi': 99, 'axis': [-2, -1]})

    if state_dict is None:
        raise KeyError("The checkpoint does not contain 'model_state_dict'.")

    # Initialize the model with transform_dic
    model = CSRNet(transform_dic=transform_dic).cuda() if use_gpu else CSRNet(transform_dic=transform_dic)

    # Handle 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load the state_dict
    try:
        model.load_state_dict(new_state_dict)
        logging.info("Model state_dict loaded successfully.")
    except RuntimeError as e:
        logging.error(f"Error loading state_dict: {e}")
        raise e

    # Optionally wrap in DataParallel if multiple GPUs are available
    if use_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Set the device
    device = torch.device(f"cuda:{gpu_device[0]}" if use_gpu and gpu_device else 'cpu')
    model.to(device)

    return model, device

def save_model(model, model_path):
    # If the model is wrapped in DataParallel, get the underlying model
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'transform_dic': model_to_save.transform_dic
    }, model_path)

# Function to save model and manage best models
def save_best_model(model, epoch, f1, saved_models, num_best_models, model_dir):
    """
    Saves the model if it's among the top N best models based on F1 score.
    Args:
    - model: The model to save.
    - epoch: The current epoch.
    - f1: The F1 score of the current model.
    - saved_models: A list tracking the saved models and their scores.
    - num_best_models: The number of best models to keep saved.
    - model_dir: The directory where models are saved.
    """    
    # Keep a list of (F1, epoch, model_filename)
    if len(saved_models) < num_best_models:
        # Save the model if there's room in the saved models list
        model_path = os.path.join(model_dir, f"model_epoch_{epoch}_f1_{f1:.4f}.pth")
        save_model(model, model_path)
        heapq.heappush(saved_models, (f1, epoch, model_path))
    else:
        # Check if the current model has a better F1 score than the worst saved model
        if f1 > saved_models[0][0]:
            _, _, old_model_path = heapq.heappop(saved_models)
            # Remove the model with the worst score
            os.remove(old_model_path)
            # Save the new model
            model_path = os.path.join(model_dir, f"model_epoch_{epoch}_f1_{f1:.4f}.pth")
            save_model(model, model_path)
            heapq.heappush(saved_models, (f1, epoch, model_path))
            logging.info(f"New model saved at {model_path}.")












    

