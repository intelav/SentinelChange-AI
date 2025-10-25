#
# Description:
# This script preprocesses multiple pairs of large Sentinel-2 scenes (.SAFE.zip) and
# converts them into sets of smaller, corresponding image tiles in PNG format.
# It uses an advanced two-stage process to generate high-quality ground truth labels:
# 1. A pre-trained ChangeStar model (on SYSU-CD) generates a coarse change proposal.
# 2. The Segment Anything Model (SAM) refines this proposal into a precise mask.
#
# This script DOES NOT run inference with your final model. It prepares the test dataset.
#
# Usage:
# 1. Ensure all paths in the CONFIG section are correct.
# 2. Download the SAM checkpoint if you haven't already.
# 3. Run the script: `python prepare_sentinel_tiles_sam.py`
#

import os
import zipfile
import glob
from tqdm import tqdm
import sys
import cv2
from collections import OrderedDict

import torch
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.io import imsave
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint, remove_module_prefix

# --- Add ChangeStar project to Python Path ---
# This allows importing modules like 'changestar_bisup'
sys.path.insert(0, '/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/')

# Import all necessary model classes so they are registered with 'ever'
from module.segmentation import Segmentation
from module.changestar import ChangeStar
from module.changestar_bisup import ChangeStarBiSup

er.registry.register_all()

# --- CONFIGURATION ---
CONFIG = {
    "base_data_dir": "/media/avaish/aiwork/satellite-work/work-changestar/data/test-data",
    "image_pairs": [
        {
            "t1": "islamabad-2023/S2A_MSIL2A_20230614T054641_N0509_R048_T43SBT_20230614T092957.SAFE.zip",
            "t2": "islamabad-2025/S2B_MSIL2A_20250608T054639_N0511_R048_T43SBT_20250608T074909.SAFE.zip",
            "name": "T43SBT"
        },
        {
            "t1": "islamabad-2023/S2B_MSIL2A_20230510T054639_N0509_R048_T43SCT_20230510T081847.SAFE.zip",
            "t2": "islamabad-2025/S2B_MSIL2A_20250608T054639_N0511_R048_T43SCT_20250608T074909.SAFE.zip",
            "name": "T43SCT"
        },
        {
            "t1": "islamabad-2023/S2B_MSIL2A_20230507T053649_N0509_R005_T43SDS_20230507T082039.SAFE.zip",
            "t2": "islamabad-2025/S2C_MSIL2A_20250809T053701_N0511_R005_T43SDS_20250809T093114.SAFE.zip",
            "name": "T43SDS"
        }
    ],
    "base_output_dir": "./prepared_test_data_islamabad",
    "bands": ['B04', 'B03', 'B02'],
    "tile_size": 1024,

    # --- Model Paths for Label Generation ---
    "changestar_config_path": "configs/sysucd/r50_farseg_changestar_finetune.py",
    "changestar_checkpoint_path": "log/finetune-SYSUCD/r50_farseg_changestar/model-12000.pth",
    "sam_checkpoint_path": "./sam_vit_h_4b8939.pth",  # <-- Make sure you have downloaded this file
    "sam_model_type": "vit_h",
}


# ---------------------

def load_changestar_model(config_path, checkpoint_path, device):
    """Loads the pre-trained ChangeStar model."""
    print(f"Loading ChangeStar model from checkpoint: {checkpoint_path}")
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
    print("ChangeStar model for labeling loaded successfully.")
    return model


def load_sam_model(checkpoint_path, model_type, device):
    """Loads the SAM model and creates a predictor."""
    print(f"Loading SAM model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: SAM checkpoint not found at {checkpoint_path}. Please download it.")
        sys.exit(1)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")
    return predictor


def preprocess_for_changestar(img_a, img_b):
    """Preprocesses a pair of numpy image tiles for the ChangeStar model."""
    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)

    # These are standard ImageNet stats used by many models
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3) * 255.0

    img_a_norm = (img_a_np - mean) / std
    img_b_norm = (img_b_np - mean) / std

    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)

    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0)
    return bi_images_tensor


def find_band_files(zip_path, bands, resolution='R10m'):
    band_files = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for filename in zf.namelist():
            if f'IMG_DATA/{resolution}' in filename and filename.endswith('.jp2'):
                for band in bands:
                    if f'_{band}_' in os.path.basename(filename):
                        band_files[band] = f'/vsizip/{zip_path}/{filename}'
    if len(band_files) != len(bands):
        raise FileNotFoundError(f"Could not find all bands {bands} in {zip_path}")
    return band_files


def read_and_stack_bands(band_file_map, target_bands):
    first_band_path = band_file_map[target_bands[0]]
    with rasterio.open(first_band_path) as src:
        profile = src.profile
        img_shape = src.shape
    stacked_img = np.zeros((img_shape[0], img_shape[1], len(target_bands)), dtype=profile['dtype'])
    for i, band_name in enumerate(target_bands):
        with rasterio.open(band_file_map[band_name]) as src:
            stacked_img[:, :, i] = src.read(1)
    profile.update(count=len(target_bands), driver='GTiff')
    return stacked_img, profile


def align_images(img_t1, profile_t1, img_t2, profile_t2):
    print("Aligning image T2 to T1...")
    aligned_img_t2 = np.zeros_like(img_t1)
    for i in range(img_t2.shape[2]):
        reproject(
            source=img_t2[:, :, i],
            destination=aligned_img_t2[:, :, i],
            src_transform=profile_t2['transform'], src_crs=profile_t2['crs'],
            dst_transform=profile_t1['transform'], dst_crs=profile_t1['crs'],
            resampling=Resampling.bilinear)
    print("Alignment complete.")
    return aligned_img_t2


def contrast_stretch(img):
    p2, p98 = np.percentile(img, (2, 98))
    stretched_img = np.clip(img, p2, p98)
    stretched_img = (stretched_img - p2) / (p98 - p2)
    return (stretched_img * 255).astype(np.uint8)


def remove_small_objects(mask, min_size=50):
    """Remove small disconnected objects from a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1
    return new_mask.astype(np.uint8)


def process_image_pair(image_t1_path, image_t2_path, output_dir, bands, tile_size, changestar_model, sam_predictor,
                       device):
    """Processes a single pair of images, aligning, tiling, and creating SAM-based labels."""
    output_dir_A = os.path.join(output_dir, 'A')
    output_dir_B = os.path.join(output_dir, 'B')
    output_dir_label = os.path.join(output_dir, 'label')
    os.makedirs(output_dir_A, exist_ok=True)
    os.makedirs(output_dir_B, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)

    print(f"Reading bands from T1: {os.path.basename(image_t1_path)}")
    band_files_t1 = find_band_files(image_t1_path, bands)
    img_t1, profile_t1 = read_and_stack_bands(band_files_t1, bands)

    print(f"Reading bands from T2: {os.path.basename(image_t2_path)}")
    band_files_t2 = find_band_files(image_t2_path, bands)
    img_t2, profile_t2 = read_and_stack_bands(band_files_t2, bands)

    img_t2_aligned = align_images(img_t1, profile_t1, img_t2, profile_t2)

    h, w, _ = img_t1.shape

    print(f"Tiling images and generating SAM labels...")
    pbar = tqdm(total=((h // tile_size) * (w // tile_size)))

    with torch.no_grad():
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end, x_end = y + tile_size, x + tile_size
                if y_end > h or x_end > w: continue

                tile_t1_raw = img_t1[y:y_end, x:x_end]
                tile_t2_raw = img_t2_aligned[y:y_end, x:x_end]

                if np.mean(tile_t1_raw) < 10 or np.mean(tile_t2_raw) < 10: continue

                tile_t1_vis = contrast_stretch(tile_t1_raw)
                tile_t2_vis = contrast_stretch(tile_t2_raw)

                # --- Generate High-Quality Label using ChangeStar + SAM ---
                input_tensor = preprocess_for_changestar(tile_t1_vis, tile_t2_vis).to(device)
                coarse_pred = changestar_model(input_tensor).sigmoid()
                coarse_map = (coarse_pred > 0.5).cpu().squeeze().numpy().astype(np.uint8)

                final_label = np.zeros_like(coarse_map)
                if coarse_map.sum() > 20:  # Only run SAM if there's a significant coarse change
                    contours, _ = cv2.findContours(coarse_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        input_boxes = torch.tensor([cv2.boundingRect(c) for c in contours], device=sam_predictor.device)
                        sam_predictor.set_image(tile_t2_vis)
                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, coarse_map.shape)
                        masks, _, _ = sam_predictor.predict_torch(
                            point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)

                        for mask in masks:
                            final_label = np.logical_or(final_label, mask.cpu().squeeze().numpy())

                final_label = remove_small_objects(final_label.astype(np.uint8)) * 255

                # --- FIX: Save tiles with pixel coordinates (y, x) in the filename ---
                tile_name = f"tile_{y}_{x}.png"
                imsave(os.path.join(output_dir_A, tile_name), tile_t1_vis)
                imsave(os.path.join(output_dir_B, tile_name), tile_t2_vis)
                imsave(os.path.join(output_dir_label, tile_name), final_label)

                pbar.update(1)

    pbar.close()
    print(f"\nSuccessfully created tile triplets (A, B, label) in '{output_dir}'.")


def main(config):
    """Main data preparation function to loop through all pairs."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load models once
    changestar_labeler = load_changestar_model(config['changestar_config_path'], config['changestar_checkpoint_path'],
                                               DEVICE)
    sam_predictor = load_sam_model(config['sam_checkpoint_path'], config['sam_model_type'], DEVICE)

    for pair in config['image_pairs']:
        print(f"\n{'=' * 20} Processing pair: {pair['name']} {'=' * 20}")

        image_t1_path = os.path.join(config['base_data_dir'], pair['t1'])
        image_t2_path = os.path.join(config['base_data_dir'], pair['t2'])
        output_dir = f"{config['base_output_dir']}_{pair['name']}"

        process_image_pair(image_t1_path, image_t2_path, output_dir, config['bands'], config['tile_size'],
                           changestar_labeler, sam_predictor, DEVICE)

    print(f"\n{'=' * 20} All pairs processed. {'=' * 20}")


if __name__ == '__main__':
    # Check for necessary libraries
    try:
        import rasterio
        from skimage.io import imsave
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:
        print(f"Error: Missing required library. {e}")
        print(
            "Please install necessary packages: pip install rasterio scikit-image tqdm segment-anything-py opencv-python")
        sys.exit(1)

    main(CONFIG)
