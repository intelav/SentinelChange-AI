#
# Description:
# This script runs inference using your fine-tuned ChangeStar model on the
# prepared Sentinel-2 test tiles. It allows you to select a specific tile
# pair and visualizes the before/after images, the ground truth label, and
# the model's prediction in a single plot for easy comparison.
#
# This workflow is ideal for manually inspecting and debugging model performance.
#
# Usage:
# 1. Update the paths and parameters in the CONFIG section below.
#    - Set the `TEST_SET_NAME` to the tile set you want to test (e.g., 'T43SBT').
#    - Set the `TILE_INDEX` to the specific tile number you want to visualize.
# 2. Run the script: `python inference_on_tiles.py`
#

import os
import sys
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint

# --- Add ChangeStar project to Python Path ---
sys.path.insert(0, '/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/')

# Import all necessary model classes so they are registered with 'ever'
from module.segmentation import Segmentation
from module.changestar import ChangeStar
from module.changestar_bisup import ChangeStarBiSup

er.registry.register_all()

# --- CONFIGURATION ---
CONFIG = {
    # --- Model to Test ---
    "model_path": "/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/log/finetune-CUSTOM-FINAL/r50_farseg_changestar/model-20000.pth",
    "config_path": "/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/configs/custom/finetune_custom_final.py",

    # --- Data to Use ---
    "base_test_data_dir": "./prepared_test_data_islamabad",
    # Choose which of the prepared tile sets to run inference on
    "test_set_name": "T43SCT",  # Options: "T43SBT", "T43SCT", "T43SDS"
    # Choose which specific tile to view
    "tile_index": 75,  # Change this to view different tiles from the set
}


# ---------------------

def load_inference_model(config_path, checkpoint_path, device):
    """Loads your fine-tuned model for inference."""
    print(f"Loading your fine-tuned model from checkpoint: {checkpoint_path}")
    cfg = ever_config.import_config(config_path)
    model = er.builder.make_model(cfg.model)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = ckpt.get(CheckPoint.MODEL, ckpt)

    # Handle checkpoints saved with DataParallel/DistributedDataParallel
    if list(model_state_dict.keys())[0].startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model_state_dict = new_state_dict

    model.load_state_dict(model_state_dict)
    model.eval()
    print("Your fine-tuned model loaded successfully.")
    return model, cfg


def preprocess_tile_pair(img_a, img_b, model_config):
    """Preprocesses a pair of PIL image tiles for your model."""
    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)

    # Use the exact normalization stats from your training config
    norm_cfg = model_config['data']['train']['params']['transforms'].transforms[2]
    mean = np.array(norm_cfg.mean, dtype=np.float32).reshape(1, 1, 6) * 255.0
    std = np.array(norm_cfg.std, dtype=np.float32).reshape(1, 1, 6) * 255.0

    # The model expects a 6-channel input, so we normalize after concatenation
    combined_img = np.concatenate([img_a_np, img_b_np], axis=2)

    # The config expects 6-channel mean/std, but we apply to 3 channels each
    mean_3ch = mean[0, 0, :3]
    std_3ch = std[0, 0, :3]

    img_a_norm = (img_a_np - mean_3ch) / std_3ch
    img_b_norm = (img_b_np - mean_3ch) / std_3ch

    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)

    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0)
    return bi_images_tensor


def main(config):
    """Main function to run inference on a single tile and visualize."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Load your fine-tuned model
    model, model_config = load_inference_model(config['config_path'], config['model_path'], DEVICE)

    # 2. Load the specific tile pair for testing
    test_dir = f"{config['base_test_data_dir']}_{config['test_set_name']}"
    dir_a = os.path.join(test_dir, 'A')
    dir_b = os.path.join(test_dir, 'B')
    dir_label = os.path.join(test_dir, 'label')

    try:
        image_files = sorted(os.listdir(dir_a))
        tile_filename = image_files[config['tile_index']]
    except IndexError:
        print(f"ERROR: Tile index {config['tile_index']} is out of bounds.")
        print(f"Please choose an index between 0 and {len(image_files) - 1}.")
        return
    except FileNotFoundError:
        print(f"ERROR: Could not find the test data directory: {test_dir}")
        print("Please make sure you have run the 'prepare_sentinel_tiles.py' script first.")
        return

    img_a_path = os.path.join(dir_a, tile_filename)
    img_b_path = os.path.join(dir_b, tile_filename)
    label_path = os.path.join(dir_label, tile_filename)

    print(f"Loading tile pair: {tile_filename}")
    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')
    label = Image.open(label_path).convert('L')

    # 3. Preprocess the tile pair and run inference
    input_tensor = preprocess_tile_pair(img_a, img_b, model_config).to(DEVICE)

    with torch.no_grad():
        prediction_logit = model(input_tensor)

    prediction_prob = prediction_logit.sigmoid()
    prediction_map = (prediction_prob > 0.5).cpu().squeeze().numpy().astype(np.uint8)

    # 4. Visualize the results
    print("Displaying results...")
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_a)
    plt.title(f'Time 1: {tile_filename}')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img_b)
    plt.title('Time 2')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(label, cmap='gray')
    plt.title('Ground Truth (from SAM)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(prediction_map, cmap='gray')
    plt.title('Model Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set rasterio cache limit
    os.environ['GDAL_CACHEMAX'] = '50000'  # in MB

    # Check for necessary libraries
    try:
        import rasterio
        import matplotlib
    except ImportError as e:
        print(f"Error: Missing required library. {e}")
        print("Please install necessary packages: pip install rasterio matplotlib")
        sys.exit(1)

    main(CONFIG)
