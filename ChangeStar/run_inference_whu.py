import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys

# Add the ChangeStar project root to the Python path
sys.path.insert(0, os.getcwd())

import ever as er
from ever.core import config as ever_config
from ever.core import to
from ever.core.checkpoint import CheckPoint, remove_module_prefix

er.registry.register_all()

# --- Configuration for Inference ---
WHU_CD_ROOT = '/media/avaish/aiwork/satellite-work/datasets/WHU-CD/building_cd_dataset/two-period-data'
DATA_SPLIT = 'test'
CHECKPOINT_PATH = 'log/bisup-LEVIRCD/r50_farseg_changestar/model-5600.pth'
IMAGE_PAIR_INDEX = 25
CONFIG_PATH = 'configs/levircd/r50_farseg_changestar_bisup.py'
# -----------------------------------

print(f"Loading configuration from: {CONFIG_PATH}")
cfg = ever_config.import_config(CONFIG_PATH)

# --- 1. Model Construction ---
model = er.builder.make_model(cfg.model)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model_state_dict = ckpt[CheckPoint.MODEL]
if hasattr(model, 'module'):
    model.module.load_state_dict(ever_config.remove_module_prefix(model_state_dict))
else:
    model.load_state_dict(remove_module_prefix(model_state_dict))

model.eval()
print("Model loaded successfully.")


# --- 2. Load and Preprocess a Single Image Pair (COMPLETELY REVISED) ---
def load_and_preprocess_image_pair_whu(root_dir, split, index):
    """
    Loads T1/T2 images and their corresponding building labels to generate
    the ground truth change mask on-the-fly.
    """
    # Define directories for images and BUILDING labels for each year
    img_a_dir = os.path.join(root_dir, '2012', 'splited_images', split, 'image')
    label_a_dir = os.path.join(root_dir, '2012', 'splited_images', split, 'label')
    img_b_dir = os.path.join(root_dir, '2016', 'splited_images', split, 'image')
    label_b_dir = os.path.join(root_dir, '2016', 'splited_images', split, 'label')

    # Get the filename for the selected index
    image_files = sorted(os.listdir(img_a_dir))
    filename = image_files[index]

    # Construct paths for all four files
    img_a_path = os.path.join(img_a_dir, filename)
    label_a_path = os.path.join(label_a_dir, filename)
    img_b_path = os.path.join(img_b_dir, filename)
    label_b_path = os.path.join(label_b_dir, filename)

    print(f"Loading Image T1:      {img_a_path}")
    print(f"Loading Image T2:      {img_b_path}")
    print(f"Loading Building Lbl T1: {label_a_path}")
    print(f"Loading Building Lbl T2: {label_b_path}")

    # Load T1 and T2 images
    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')

    # Load T1 and T2 BUILDING labels
    label_a = Image.open(label_a_path).convert('L')
    label_b = Image.open(label_b_path).convert('L')

    # --- Generate Ground Truth Change Label ---
    label_a_np = np.array(label_a) / 255.0 # Normalize to 0/1
    label_b_np = np.array(label_b) / 255.0 # Normalize to 0/1
    # A change is where the building labels are different
    gt_change_label_np = np.where(label_a_np != label_b_np, 1.0, 0.0).astype(np.float32)
    # Convert the generated numpy change label back to a PIL Image for visualization
    gt_change_label_pil = Image.fromarray((gt_change_label_np * 255).astype(np.uint8))
    print("Ground truth change mask generated successfully.")

    # --- Preprocess Images for Model Input ---
    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    img_a_norm = (img_a_np - mean) / std
    img_b_norm = (img_b_np - mean) / std

    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)

    # Create tensor for model input
    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0)
    # Create tensor from the GENERATED ground truth change label
    gt_change_label_tensor = torch.from_numpy(gt_change_label_np).unsqueeze(0).unsqueeze(0)

    return bi_images_tensor, gt_change_label_tensor, img_a, img_b, gt_change_label_pil

# --- Main Execution ---
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from WHU-CD '{DATA_SPLIT}' set...")
bi_images_tensor, gt_label_tensor, original_img_a, original_img_b, original_label = \
    load_and_preprocess_image_pair_whu(WHU_CD_ROOT, DATA_SPLIT, IMAGE_PAIR_INDEX)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bi_images_tensor = bi_images_tensor.to(device)
print("Image pair loaded and preprocessed.")

print("Performing inference...")
with torch.no_grad():
    predictions = model(bi_images_tensor)

predicted_change_prob = predictions.sigmoid()
predicted_change_map = (predicted_change_prob > 0.5).float()

predicted_change_map_np = predicted_change_map.cpu().squeeze().numpy()
gt_label_np = gt_label_tensor.cpu().squeeze().numpy()
print("Inference complete.")

print("Displaying results...")
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(original_img_a)
plt.title(f'Time 1 (2012) - Index: {IMAGE_PAIR_INDEX}')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(original_img_b)
plt.title('Time 2 (2016)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gt_label_np, cmap='gray')
plt.title('Ground Truth Change')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(predicted_change_map_np, cmap='gray')
plt.title('Predicted Change')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nInference and visualization complete for the selected WHU-CD image pair.")