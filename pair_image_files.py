import os
import re
from collections import defaultdict
import json
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_tile_id_from_path(path):
    """Extracts the MGRS tile ID (e.g., T43SFS) from a Sentinel-2 .SAFE path."""
    match = re.search(r'_T([A-Z0-9]{5})_', path)
    if match:
        return match.group(1)
    logging.warning(f"Could not extract tile ID from path: {path}")
    return None


def pair_sentinel_images(data_root):
    """
    Pairs Sentinel-2 images from different years based on their MGRS tile ID.
    """
    images_by_tile = defaultdict(lambda: defaultdict(list))

    logging.info(f"Starting to scan for .SAFE directories in: {data_root}")
    # Walk through the directory to find all .SAFE directories
    for root, dirs, _ in os.walk(data_root):
        for dir_name in dirs:
            if dir_name.endswith('.SAFE'):
                full_path = os.path.join(root, dir_name)
                tile_id = get_tile_id_from_path(dir_name)

                if tile_id:
                    # Extract terrain and year from the path
                    parts = root.replace(data_root, '').strip(os.sep).split(os.sep)
                    if len(parts) == 1:
                        terrain_year = parts[0]
                        # Attempt to parse terrain and year
                        match = re.match(r'([a-zA-Z]+)-(\d{4})', terrain_year)
                        if match:
                            terrain = match.group(1)
                            year = match.group(2)
                            logging.info(
                                f"Found image: Tile={tile_id}, Year={year}, Terrain={terrain}, Path={full_path}")
                            images_by_tile[tile_id][year].append(full_path)
                        else:
                            logging.warning(f"Could not parse terrain and year from folder: {terrain_year}")

    # Create pairs
    logging.info("Finished scanning. Now creating image pairs.")
    image_pairs = []
    for tile_id, year_files in images_by_tile.items():
        sorted_years = sorted(year_files.keys())
        if len(sorted_years) >= 2:
            logging.info(f"Processing tile {tile_id} with images from years: {sorted_years}")
            # Pair images from different years
            for i in range(len(sorted_years)):
                for j in range(i + 1, len(sorted_years)):
                    year1 = sorted_years[i]
                    year2 = sorted_years[j]

                    logging.info(f"  - Pairing year {year1} with year {year2} for tile {tile_id}")
                    # Create all combinations of pairs between the two years
                    for img1_path in year_files[year1]:
                        for img2_path in year_files[year2]:
                            pair_info = {
                                'tile_id': tile_id,
                                'image1_path': img1_path,
                                'image2_path': img2_path,
                                'year1': year1,
                                'year2': year2,
                                'terrain': os.path.basename(os.path.dirname(img1_path)).split('-')[0]
                            }
                            image_pairs.append(pair_info)
                            logging.debug(f"    Created pair: {json.dumps(pair_info)}")

    return image_pairs


if __name__ == '__main__':
    DATA_ROOT = './'
    pairs = pair_sentinel_images(DATA_ROOT)

    logging.info(f"Found a total of {len(pairs)} image pairs.")

    # Save the pairs to a JSON file for the next step
    output_json_path = os.path.join(DATA_ROOT, 'image_pairs.json')
    try:
        with open(output_json_path, 'w') as f:
            json.dump(pairs, f, indent=4)
        logging.info(f"Image pairs successfully saved to {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to save image pairs to JSON file: {e}")

    # Print a few examples
    logging.info("--- Example Pairs ---")
    for pair in pairs[:5]:
        print(json.dumps(pair, indent=2))
    logging.info("---------------------")
