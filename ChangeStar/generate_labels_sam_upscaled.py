import torch
import numpy as np
from PIL import Image
import os
import sys
from tqdm import tqdm
import cv2
from segment_anything import sam_model_registry, SamPredictor
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the ChangeStar project root to the Python path
sys.path.insert(0, '/media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/')  # Adjust if necessary

import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint, remove_module_prefix

# This assumes the model definitions are available in the path
er.registry.register_all()


def load_changestar_model(config_path, checkpoint_path, device):
    """Loads the pre-trained ChangeStar model."""
    logging.info(f"Loading ChangeStar model from checkpoint: {checkpoint_path}")
    cfg = ever_config.import_config(config_path)
    model = er.builder.make_model(cfg.model)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = ckpt.get(CheckPoint.MODEL, ckpt)
    model.load_state_dict(remove_module_prefix(model_state_dict))
    model.eval()
    logging.info("ChangeStar model loaded successfully.")
    return model


def load_sam_model(checkpoint_path, model_type, device):
    """Loads the SAM model and creates a predictor."""
    logging.info(f"Loading SAM model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logging.error(f"SAM checkpoint not found at {checkpoint_path}. Please download it.")
        sys.exit(1)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    logging.info("SAM model loaded successfully.")
    return predictor


def preprocess_pair_for_changestar(img_a_path, img_b_path):
    """Preprocesses a pair of images for the ChangeStar model."""
    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')

    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    img_a_norm = (img_a_np - mean) / std
    img_b_norm = (img_b_np - mean) / std

    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)

    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0)
    return bi_images_tensor


def remove_small_objects(mask, min_size=100):
    """Remove small disconnected objects from a binary mask."""
    # Find all connected components (objects)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create an output image that's the same size as the input
    new_mask = np.zeros_like(mask)

    # Iterate through each component, keeping only the large ones
    for i in range(1, num_labels):  # start from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1

    return new_mask.astype(np.uint8)


def generate_sam_labels(changestar_model, sam_predictor, processed_data_dir, device):
    """
    Generates high-quality change labels using ChangeStar for proposals
    and SAM for precise segmentation on upscaled images.
    """
    dir_a = os.path.join(processed_data_dir, 'train', 'A')
    dir_b = os.path.join(processed_data_dir, 'train', 'B')
    label_dir = os.path.join(processed_data_dir, 'train', 'label')
    os.makedirs(label_dir, exist_ok=True)

    image_files = sorted(os.listdir(dir_a))

    change_detected_count = 0
    no_change_count = 0
    files_with_change = []
    files_without_change = []

    logging.info(f"Starting to generate SAM-based labels for {len(image_files)} upscaled image pairs...")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Generating Upscaled Labels"):
            logging.info(f"--- Processing file: {filename} ---")
            img_a_path = os.path.join(dir_a, filename)
            img_b_path = os.path.join(dir_b, filename)

            # 1. Get coarse change map from ChangeStar
            logging.debug("Step 1: Getting coarse change map from ChangeStar.")
            input_tensor = preprocess_pair_for_changestar(img_a_path, img_b_path).to(device)
            coarse_prediction = changestar_model(input_tensor).sigmoid()
            coarse_map = (coarse_prediction > 0.5).cpu().squeeze().numpy().astype(np.uint8)
            logging.info(f"Coarse map generated. Total changed pixels: {coarse_map.sum()}")

            if coarse_map.sum() < 50:  # Increased threshold for higher resolution
                logging.info("No significant change detected by ChangeStar. Saving empty label.")
                # UPDATED: Create a 4096x4096 empty label for the upscaled images
                empty_label = Image.fromarray(np.zeros((4096, 4096), dtype=np.uint8), 'L')
                label_path = os.path.join(label_dir, filename)
                if not os.path.exists(label_path):
                    empty_label.save(label_path)
                no_change_count += 1
                files_without_change.append(filename)
                continue

            # 2. Find contours to create bounding box prompts
            logging.debug("Step 2: Finding contours for SAM prompts.")
            contours, _ = cv2.findContours(coarse_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logging.warning("ChangeStar detected pixels but no contours found. Saving empty label.")
                empty_label = Image.fromarray(np.zeros_like(coarse_map), 'L')
                empty_label.save(os.path.join(label_dir, filename))
                no_change_count += 1
                files_without_change.append(filename)
                continue

            logging.info(f"Found {len(contours)} potential change regions (contours).")
            input_boxes = torch.tensor([cv2.boundingRect(c) for c in contours], device=sam_predictor.device)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, coarse_map.shape)

            # 3. Use SAM to get precise masks
            logging.debug("Step 3: Using SAM to get precise masks.")
            image_b_for_sam = cv2.imread(img_b_path)
            if image_b_for_sam is None:
                logging.error(f"Failed to read image for SAM: {img_b_path}")
                continue
            image_b_for_sam = cv2.cvtColor(image_b_for_sam, cv2.COLOR_BGR2RGB)

            sam_predictor.set_image(image_b_for_sam)
            logging.debug("SAM image set.")

            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            logging.info(f"SAM generated {len(masks)} masks.")

            # 4. Combine masks into a single label
            logging.debug("Step 4: Combining masks into a final label.")
            final_label = np.zeros_like(coarse_map, dtype=np.uint8)
            for mask in masks:
                final_label = np.logical_or(final_label, mask.cpu().squeeze().numpy()).astype(np.uint8)

            # UPDATED: Increased min_size for higher resolution images
            cleaned_label = remove_small_objects(final_label, min_size=100)
            logging.info(
                f"Final label created. Pixels before cleaning: {final_label.sum()}, Pixels after cleaning: {cleaned_label.sum()}")

            if cleaned_label.sum() > 0:
                change_detected_count += 1
                files_with_change.append(filename)
            else:
                no_change_count += 1
                files_without_change.append(filename)

            label_img = Image.fromarray(cleaned_label * 255, 'L')
            label_img.save(os.path.join(label_dir, filename))
            logging.info(f"Successfully saved label for {filename}")

    logging.info("--- Finished generating all SAM-based labels. ---")

    logging.info("--- Change Detection Summary ---")
    logging.info(f"Total pairs processed: {len(image_files)}")
    logging.info(f"Pairs with detected change: {change_detected_count}")
    logging.info(f"Pairs with no change: {no_change_count}")
    logging.info("--------------------------------")

    summary_path = os.path.join(processed_data_dir, 'change_detection_summary_upscaled.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write("--- Change Detection Summary (Upscaled) ---\n")
            f.write(f"Total pairs processed: {len(image_files)}\n")
            f.write(f"Pairs with detected change: {change_detected_count}\n")
            f.write(f"Pairs with no change: {no_change_count}\n")
            f.write("\n--- Files with Change ---\n")
            for fname in files_with_change:
                f.write(f"{fname}\n")
            f.write("\n--- Files with No Change ---\n")
            for fname in files_without_change:
                f.write(f"{fname}\n")
        logging.info(f"Summary report saved to: {summary_path}")
    except Exception as e:
        logging.error(f"Failed to write summary report: {e}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    CHANGESTAR_CONFIG_PATH = 'configs/sysucd/r50_farseg_changestar_finetune.py'
    CHANGESTAR_CHECKPOINT_PATH = 'log/finetune-SYSUCD/r50_farseg_changestar/model-12000.pth'

    SAM_CHECKPOINT_PATH = './sam_vit_h_4b8939.pth'
    SAM_MODEL_TYPE = 'vit_h'

    # UPDATED: Point to the new directory with upscaled images
    PROCESSED_DATA_DIR = './MyCustomCD_Dataset_Upscaled'
    # ---

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {DEVICE}")

    changestar_model = load_changestar_model(CHANGESTAR_CONFIG_PATH, CHANGESTAR_CHECKPOINT_PATH, DEVICE)
    sam_predictor = load_sam_model(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, DEVICE)

    generate_sam_labels(changestar_model, sam_predictor, PROCESSED_DATA_DIR, DEVICE)
