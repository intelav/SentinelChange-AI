#
# Description:
# This script runs inference using a fine-tuned ChangeStar model on prepared
# Sentinel-2 test tiles. It processes all tiles in a specified directory and
# generates output files in the format required by the problem statement:
#   1. A georeferenced raster mask (GeoTIFF) with 1 for change, 0 for no change.
#   2. A corresponding vector file (Shapefile) of the change polygons.
#   3. Files are named using the tile's center coordinates.
#   4. The MD5 hash of the model checkpoint is calculated and printed.
#
# Usage:
# 1. Update paths in the CONFIG section.
# 2. Ensure your prepared tiles have filenames ending in '_r_c.png' (e.g., 'tile_1024_512.png').
# 3. Run the script: `python generate_submission.py`
#

import os
import sys
import hashlib
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint
from tqdm import tqdm

# Geospatial libraries
import rasterio
from rasterio.transform import Affine
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping

# --- Add ChangeStar project to Python Path ---
sys.path.insert(0, '/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/')

# Import all necessary model classes so they are registered with 'ever'
from module.segmentation import Segmentation
from module.changestar import ChangeStar
from module.changestar_bisup import ChangeStarBiSup

er.registry.register_all()

# --- CONFIGURATION ---
CONFIG = {
    # --- Model to Use ---
    "model_path": "/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/log/finetune-CUSTOM-FINAL/r50_farseg_changestar/model-20000.pth",
    "config_path": "/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/configs/custom/finetune_custom_final.py",

    # --- Data to Process ---
    "base_test_data_dir": "./prepared_test_data_islamabad",
    "test_set_name": "T43SBT",  # Options: "T43SBT", "T43SCT", "T43SDS"

    # --- Original Scene for Georeferencing ---
    # The script needs the original .SAFE.zip file to get the coordinate system for the tiles.
    "original_scene_path": "/media/avaish/aiwork/satellite-work/work-changestar/data/test-data/islamabad-2023/S2A_MSIL2A_20230614T054641_N0509_R048_T43SBT_20230614T092957.SAFE.zip",

    # --- Output Directory ---
    "submission_output_dir": "./submission_files"
}


# ---------------------

def get_md5_hash(file_path):
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_inference_model(config_path, checkpoint_path, device):
    """Loads your fine-tuned model for inference."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    cfg = ever_config.import_config(config_path)
    model = er.builder.make_model(cfg.model)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = ckpt.get(CheckPoint.MODEL, ckpt)

    if list(model_state_dict.keys())[0].startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model_state_dict = new_state_dict

    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model loaded successfully.")
    return model, cfg


def preprocess_tile_pair(img_a, img_b, model_config):
    """Preprocesses a pair of PIL image tiles for your model."""
    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)

    norm_cfg = model_config['data']['train']['params']['transforms'].transforms[2]
    mean = np.array(norm_cfg.mean, dtype=np.float32).reshape(1, 1, 6) * 255.0
    std = np.array(norm_cfg.std, dtype=np.float32).reshape(1, 1, 6) * 255.0

    mean_3ch = mean[0, 0, :3]
    std_3ch = std[0, 0, :3]

    img_a_norm = (img_a_np - mean_3ch) / std_3ch
    img_b_norm = (img_b_np - mean_3ch) / std_3ch

    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)

    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0)
    return bi_images_tensor


def get_georef_profile(original_scene_path):
    """Reads the georeferencing info from the original Sentinel-2 scene."""
    # Find any 10m band file to read the profile
    import zipfile
    with zipfile.ZipFile(original_scene_path, 'r') as zf:
        for filename in zf.namelist():
            if 'IMG_DATA/R10m' in filename and filename.endswith('.jp2'):
                band_path_in_zip = f'/vsizip/{original_scene_path}/{filename}'
                with rasterio.open(band_path_in_zip) as src:
                    return src.profile, src.crs, src.transform
    raise FileNotFoundError("Could not find a 10m band file in the original scene zip.")


def main(config):
    """Main function to run inference on all tiles and generate submission files."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Calculate and print model hash
    model_hash = get_md5_hash(config['model_path'])
    print(f"\n{'=' * 20} MODEL MD5 HASH {'=' * 20}")
    print(model_hash)
    print(f"{'=' * 58}\n")

    # 2. Load your fine-tuned model
    model, model_config = load_inference_model(config['config_path'], config['model_path'], DEVICE)

    # 3. Get base georeferencing from the original scene
    print("Reading georeferencing info from original scene...")
    full_profile, crs, full_transform = get_georef_profile(config['original_scene_path'])
    tile_size = model_config['data']['train']['params']['transforms'][1].height

    # 4. Prepare directories
    test_dir = f"{config['base_test_data_dir']}_{config['test_set_name']}"
    output_dir = os.path.join(config['submission_output_dir'], config['test_set_name'])
    os.makedirs(output_dir, exist_ok=True)

    dir_a = os.path.join(test_dir, 'A')
    dir_b = os.path.join(test_dir, 'B')
    image_files = sorted(os.listdir(dir_a))

    print(f"\nProcessing {len(image_files)} tiles from {test_dir}...")
    for tile_filename in tqdm(image_files, desc="Generating Submission Files"):
        img_a_path = os.path.join(dir_a, tile_filename)
        img_b_path = os.path.join(dir_b, tile_filename)

        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')

        # 5. Run Inference
        input_tensor = preprocess_tile_pair(img_a, img_b, model_config).to(DEVICE)
        with torch.no_grad():
            prediction_logit = model(input_tensor)

        prediction_map = (prediction_logit.sigmoid() > 0.5).cpu().squeeze().numpy().astype(np.uint8)

        # 6. Generate Outputs
        if prediction_map.sum() > 0:  # Only save if change is detected
            # --- FIX: Robustly parse row and column from filename ---
            # This logic now expects filenames to end with '_r_c.png', e.g., '..._1024_512.png'
            # It will safely skip files that do not match this format.
            try:
                parts = os.path.splitext(tile_filename)[0].split('_')
                r, c = int(parts[-2]), int(parts[-1])
            except (ValueError, IndexError):
                print(f"\n[Warning] Could not parse coordinates from filename: {tile_filename}. Skipping.")
                continue

            # Calculate tile-specific georeferencing
            tile_transform = rasterio.windows.transform(rasterio.windows.Window(c, r, tile_size, tile_size),
                                                        full_transform)

            # Get center coordinates for filename
            lon, lat = tile_transform * (tile_size / 2, tile_size / 2)
            base_filename = f"Change_Mask_{lat:.4f}_{lon:.4f}"

            # --- Save GeoTIFF ---
            tif_path = os.path.join(output_dir, f"{base_filename}.tif")
            profile = {
                'driver': 'GTiff', 'height': tile_size, 'width': tile_size,
                'count': 1, 'dtype': rasterio.uint8, 'crs': crs,
                'transform': tile_transform, 'nodata': 0
            }
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(prediction_map, 1)

            # --- Save Shapefile ---
            shp_path = os.path.join(output_dir, f"{base_filename}.shp")
            mask = prediction_map == 1
            shapes_gen = shapes(prediction_map, mask=mask, transform=tile_transform)

            schema = {'geometry': 'Polygon', 'properties': {'change': 'int'}}
            with fiona.open(shp_path, 'w', 'ESRI Shapefile', schema, crs=crs) as collection:
                for poly, value in shapes_gen:
                    if value == 1:
                        collection.write({'geometry': poly, 'properties': {'change': 1}})

    print(f"\nProcessing complete. Submission files saved in: {output_dir}")


if __name__ == '__main__':
    os.environ['GDAL_CACHEMAX'] = '50000'  # in MB
    try:
        import rasterio, fiona, shapely
    except ImportError as e:
        print(f"Error: Missing required library. {e}")
        print("Please install necessary packages: pip install rasterio fiona shapely")
        sys.exit(1)

    main(CONFIG)
