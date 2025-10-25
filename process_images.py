import os
import json
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
from tqdm import tqdm
import fnmatch
import logging
import math

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_band_file(safe_dir_path, band_name):
    """Finds the path to a specific band file within a .SAFE directory."""
    # Handle the common case where unzipping creates a nested folder with the same name
    search_path = safe_dir_path
    potential_nested_dir = os.path.join(safe_dir_path, os.path.basename(safe_dir_path))
    if os.path.isdir(potential_nested_dir):
        logging.debug(f"Found nested directory structure. Searching in: {potential_nested_dir}")
        search_path = potential_nested_dir

    granule_dir = os.path.join(search_path, 'GRANULE')
    if not os.path.exists(granule_dir):
        logging.warning(f"GRANULE directory not found in {search_path}")
        return None

    try:
        # Sentinel-2 data has one or more tile folders inside GRANULE
        tile_dirs = [d for d in os.listdir(granule_dir) if os.path.isdir(os.path.join(granule_dir, d))]
        if not tile_dirs:
            logging.warning(f"No tile directory found in {granule_dir}")
            return None

        # Search within each tile directory found
        for tile_dir_name in tile_dirs:
            img_data_dir = os.path.join(granule_dir, tile_dir_name, 'IMG_DATA')
            if not os.path.exists(img_data_dir):
                logging.warning(f"IMG_DATA directory not found in {os.path.join(granule_dir, tile_dir_name)}")
                continue

            # Sentinel-2 Level-2A data has bands in a resolution subdirectory (e.g., R10m)
            for root, _, files in os.walk(img_data_dir):
                for file in files:
                    # FIX: Make the search flexible to account for resolution (e.g., _10m, _20m) in the filename.
                    if f'_{band_name}_' in file and file.endswith('.jp2'):
                        logging.debug(f"Found band {band_name} at {os.path.join(root, file)}")
                        return os.path.join(root, file)
    except Exception as e:
        logging.error(f"Error finding band {band_name} in {search_path}: {e}")

    logging.warning(f"Band {band_name} not found in {search_path}")
    return None


def create_full_res_rgb_image(safe_dir_path, terrain_type):
    """Creates a full-resolution RGB image from Sentinel-2 bands without resizing."""
    logging.info(f"Processing SAFE directory for full-resolution RGB: {safe_dir_path}")
    bands_to_read = ['B04', 'B03', 'B02']  # Red, Green, Blue
    band_files = [find_band_file(safe_dir_path, band) for band in bands_to_read]

    if any(f is None for f in band_files):
        logging.error(f"Could not find all required bands for {safe_dir_path}. Skipping.")
        return None

    try:
        logging.debug(f"Reading bands: {band_files}")
        # Open band files but don't read yet, to get metadata
        band_datasets = [rasterio.open(f) for f in band_files]

        # Assume all bands have the same dimensions
        width = band_datasets[0].width
        height = band_datasets[0].height

        # Read the full data
        bands_data = [ds.read(1) for ds in band_datasets]

        rgb_image = np.stack(bands_data, axis=-1)
        logging.debug(f"Stacked bands. Full-res image shape: {rgb_image.shape}")

        # Apply normalization
        p_low, p_high = 2, 98
        img_stretched = np.zeros_like(rgb_image, dtype=np.uint8)

        channel_names = ['Red', 'Green', 'Blue']
        for i in range(rgb_image.shape[2]):
            channel_data = rgb_image[:, :, i]
            # Avoid no-data values (often 0) in percentile calculation
            valid_pixels = channel_data[channel_data > 0]
            if len(valid_pixels) == 0:
                logging.warning(f"Channel {channel_names[i]} contains no valid data. Skipping stretch.")
                continue

            v_min, v_max = np.percentile(valid_pixels, (p_low, p_high))

            # FIX: For snow, use a more aggressive stretch to make snow appear white
            if terrain_type == 'snow':
                # Use a higher percentile for the max value to capture bright snow
                p_high_snow = 99.8
                v_max = np.percentile(valid_pixels, p_high_snow)
                logging.info(f"Applying special snow handling. Using {p_high_snow}th percentile for max value.")

            # ADDED LOGS: Print the calculated min/max values for stretching
            logging.info(
                f"Color Stretch ({channel_names[i]}): Raw Min={v_min:.2f}, Raw Max={v_max:.2f}. These values will be mapped to 0 and 255.")

            # Scale to 0-255
            band_stretched = np.clip((channel_data - v_min) / (v_max - v_min), 0, 1) * 255
            img_stretched[:, :, i] = band_stretched.astype(np.uint8)

        logging.debug("Applied percentile stretch for normalization.")

        return Image.fromarray(img_stretched)

    except Exception as e:
        logging.error(f"Failed to process {safe_dir_path}: {e}")
        return None


def tile_and_save_image(full_res_image, base_filename, output_dir, tile_size=(1024, 1024)):
    """Tiles the full-resolution image and saves the tiles."""
    width, height = full_res_image.size
    tile_w, tile_h = tile_size

    for r in range(0, height, tile_h):
        for c in range(0, width, tile_w):
            # Define the box for cropping
            box = (c, r, c + tile_w, r + tile_h)

            # Crop the image to get the tile
            tile = full_res_image.crop(box)

            # Create a unique filename for each tile
            tile_filename = f"{base_filename}_tile_{r}_{c}.png"
            tile_path = os.path.join(output_dir, tile_filename)

            # Save the tile
            tile.save(tile_path)
            logging.debug(f"Saved tile: {tile_path}")


def process_image_pairs(pairs_json_path, output_dir, tile_size=(1024, 1024)):
    """
    Processes all image pairs, creating full-res images, tiling them, and saving them.
    """
    try:
        with open(pairs_json_path, 'r') as f:
            pairs = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The pairs file was not found at {pairs_json_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {pairs_json_path}")
        return

    dir_a = os.path.join(output_dir, 'train', 'A')
    dir_b = os.path.join(output_dir, 'train', 'B')
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)
    logging.info(f"Output directories created at {os.path.join(output_dir, 'train')}")

    logging.info(f"Starting to process {len(pairs)} pairs. Tile size: {tile_size}")
    for i, pair in enumerate(tqdm(pairs, desc="Processing pairs")):
        logging.info(f"--- Processing Pair {i + 1}/{len(pairs)} ---")
        img1_path = pair['image1_path']
        img2_path = pair['image2_path']
        terrain = pair['terrain']

        # Create full-resolution RGB images first, passing the terrain type
        rgb1_full = create_full_res_rgb_image(img1_path, terrain)
        rgb2_full = create_full_res_rgb_image(img2_path, terrain)

        if rgb1_full and rgb2_full:
            # Create a base filename for the pair
            base_filename = f"{pair['terrain']}_{pair['tile_id']}_{pair['year1']}_{pair['year2']}_{i}"

            # Tile and save both images
            logging.info(f"Tiling image 1 for pair {i + 1}...")
            tile_and_save_image(rgb1_full, base_filename, dir_a, tile_size)

            logging.info(f"Tiling image 2 for pair {i + 1}...")
            tile_and_save_image(rgb2_full, base_filename, dir_b, tile_size)

            logging.info(f"Successfully tiled and saved pair {i + 1}")
        else:
            logging.error(f"Skipping pair {i + 1} due to an error in creating one or both full-res RGB images.")

    logging.info("--- Finished processing all image pairs. ---")


if __name__ == '__main__':
    PAIRS_JSON = './image_pairs.json'
    PROCESSED_DATA_DIR = './MyCustomCD_Dataset'
    TILE_SIZE = (1024, 1024)

    process_image_pairs(PAIRS_JSON, PROCESSED_DATA_DIR, TILE_SIZE)
