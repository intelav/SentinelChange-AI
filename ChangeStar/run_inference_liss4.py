import os
import glob
import zipfile
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import torch
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Setup Project Path ---
# This ensures the script can find the custom modules
sys.path.insert(0, os.getcwd())

import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint, remove_module_prefix
# import module.models.changestar

er.registry.register_all()

# --- CONFIGURATION ---
# 1. Path to the root directory containing your LISS-4 zip files
LISS4_DATA_ROOT = '../data-bhoonidhi/urban-2025'

# 2. Path to your newly trained model and its config file
CHECKPOINT_PATH = 'log/finetune-CUSTOM-FINAL/r50_farseg_changestar/model-20000.pth'
CONFIG_PATH = 'configs/custom/finetune_custom_final.py'

# 3. Directory to save the output change masks
OUTPUT_DIR = './liss4_inference_results'

# 4. The size of tiles to process. Should match your training patch size.
TILE_SIZE = (512, 512)


# ---------------------

def unzip_and_find_tiffs(root_dir):
    """Unzips all .zip files and returns a list of .tif files."""
    # Unzip all files first
    for zip_path in glob.glob(os.path.join(root_dir, '*.zip')):
        extract_path = zip_path.rsplit('.zip', 1)[0]
        if not os.path.exists(extract_path):
            print(f"Unzipping {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    # Find all .tif files in the unzipped directories
    tiff_files = glob.glob(os.path.join(root_dir, '**', '*.tif'), recursive=True)
    print(f"Found {len(tiff_files)} .tif files.")
    return tiff_files


def pair_liss4_images(tiff_files):
    """Pairs LISS-4 images based on their unique identifier."""
    images_by_id = defaultdict(list)
    for fp in tiff_files:
        # Assumes the unique ID is part of the filename before the .tif extension
        # e.g., '...GTDA' in '...GTDA.tif'
        unique_id = os.path.basename(fp).split('.')[0][-4:]
        images_by_id[unique_id].append(fp)

    pairs = []
    for unique_id, files in images_by_id.items():
        if len(files) >= 2:
            # Simple pairing: take the first two found for each ID
            # A more robust method would sort by date from the filename
            pairs.append((files[0], files[1]))
            print(f"Paired: {os.path.basename(files[0])} and {os.path.basename(files[1])}")
    return pairs


def load_model(config_path, checkpoint_path, device):
    """Loads the fine-tuned ChangeStar model."""
    print("Loading fine-tuned model...")
    cfg = ever_config.import_config(config_path)
    model = er.builder.make_model(cfg.model)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = ckpt.get(CheckPoint.MODEL, ckpt)
    model.load_state_dict(remove_module_prefix(model_state_dict))
    model.eval()
    print("Model loaded successfully.")
    return model


def preprocess_tile(tile_data):
    """Preprocesses a single tile for the model."""
    # LISS-4 has 3 bands. We assume they are in R, G, B order.
    # If they are G, R, NIR, you might need to swap them: tile_data[[1, 0, 2], :, :]
    img_np = np.array(tile_data, dtype=np.float32).transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255

    img_norm = (img_np - mean) / std

    return torch.from_numpy(img_norm).permute(2, 0, 1)


def run_inference():
    """Main function to run the full inference pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 2. Find and Pair Data
    tiff_files = unzip_and_find_tiffs(LISS4_DATA_ROOT)
    image_pairs = pair_liss4_images(tiff_files)

    if not image_pairs:
        print("No image pairs found. Exiting.")
        return

    # 3. Process each pair
    with torch.no_grad():
        for i, (path_a, path_b) in enumerate(image_pairs):
            print(f"\n--- Processing Pair {i + 1}/{len(image_pairs)} ---")

            with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
                if src_a.count < 3 or src_b.count < 3:
                    print(f"Warning: Skipping pair, not enough bands. A: {src_a.count}, B: {src_b.count}")
                    continue

                width, height = src_a.width, src_a.height
                tile_w, tile_h = TILE_SIZE

                for r in tqdm(range(0, height, tile_h), desc=f"Processing {os.path.basename(path_a)}"):
                    for c in range(0, width, tile_w):
                        window = Window(c, r, tile_w, tile_h)

                        # Read tiles from both images
                        tile_a = src_a.read(window=window)
                        tile_b = src_b.read(window=window)

                        # Skip if tile is smaller than expected (at the edges)
                        if tile_a.shape[1] != tile_h or tile_a.shape[2] != tile_w:
                            continue

                        # Preprocess tiles
                        t1_tensor = preprocess_tile(tile_a[:3, :, :])  # Use first 3 bands
                        t2_tensor = preprocess_tile(tile_b[:3, :, :])  # Use first 3 bands

                        # Combine and send to GPU
                        input_tensor = torch.cat([t1_tensor, t2_tensor], dim=0).unsqueeze(0).to(device)

                        # Run inference
                        prediction = model(input_tensor).sigmoid()
                        change_mask = (prediction > 0.5).cpu().squeeze().numpy().astype(np.uint8)

                        # Save output mask as GeoTIFF
                        if change_mask.sum() > 0:  # Only save if change is detected
                            tile_transform = src_a.window_transform(window)
                            profile = src_a.profile
                            profile.update({
                                'dtype': 'uint8',
                                'count': 1,
                                'compress': 'lzw',
                                'transform': tile_transform,
                                'width': tile_w,
                                'height': tile_h
                            })

                            output_filename = f"change_{os.path.basename(path_a).split('.')[0]}_tile_{r}_{c}.tif"
                            output_path = os.path.join(OUTPUT_DIR, output_filename)

                            with rasterio.open(output_path, 'w', **profile) as dst:
                                dst.write(change_mask, 1)

    print("\n--- Inference Complete ---")
    print(f"Change masks saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_inference()
