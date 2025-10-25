import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys
import glob  # For listing files based on patterns

# Add the ChangeStar project root to the Python path
sys.path.insert(0, os.getcwd())

import ever as er
from ever.core import config as ever_config
from ever.core import to
from ever.core.checkpoint import CheckPoint, remove_module_prefix

er.registry.register_all()  # Crucial for model building

# --- Configuration for Inference ---
# Path to your specific checkpoint file (trained on xView2)
CHECKPOINT_PATH = 'log/changestar_sisup/r50_farseg_changemixin_symmetry/model-22000.pth'

# Root directory of your xView2 dataset (test split)
XVIEW2_TEST_ROOT = '/media/avaish/aiwork/satellite-work/datasets/xview2/test'

# Index of the image pair you want to use from the test set
# The 'pair' here refers to a 'pre_disaster.png' image and its '_pre_disaster_target.png' mask
IMAGE_PAIR_INDEX = 4  # Change this to test different image pairs from 0 up to (total pre_disaster images - 1)

# --- Model Configuration (should match your training config) ---
CONFIG_PATH = 'configs/trainxView2/r50_farseg_changemixin_symmetry.py'

# --- Image Preprocessing Parameters ---
# These mean/std values are from standard.py for xView2 training (3-channel input)
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model was trained on 512x512 crops
TARGET_H, TARGET_W = 512, 512

# -----------------------------------

print(f"Loading configuration from: {CONFIG_PATH}")
cfg = ever_config.import_config(CONFIG_PATH)

# --- 1. Model Construction ---
print("Constructing model architecture...")
model = er.builder.make_model(cfg.model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

model_state_dict = ckpt[CheckPoint.MODEL]
if hasattr(model, 'module'):
    model.module.load_state_dict(remove_module_prefix(model_state_dict))
else:
    model.load_state_dict(remove_module_prefix(model_state_dict))

model.eval()  # Set model to evaluation mode
print("Model loaded successfully.")


# --- 2. Load and Preprocess a Single xView2 Image Pair ---
def load_and_preprocess_xview2_pair(root_dir, index, target_h, target_w, norm_mean, norm_std):
    images_dir = os.path.join(root_dir, 'images')
    targets_dir = os.path.join(root_dir, 'targets')  # xView2 test set has a targets folder

    # Find all 'pre_disaster_target.png' files
    pre_target_files = sorted(glob.glob(os.path.join(targets_dir, '*_pre_disaster_target.png')))

    if index >= len(pre_target_files):
        raise IndexError(
            f"Index {index} out of bounds for xView2 pre-disaster targets. Max index is {len(pre_target_files) - 1}.")

    target_path = pre_target_files[index]

    # Derive corresponding image filename (e.g., from '_pre_disaster_target.png' to '_pre_disaster.png')
    target_filename = os.path.basename(target_path)
    image_filename = target_filename.replace('_target.png', '.png')
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Corresponding image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    target = Image.open(target_path).convert('L')  # Ground truth building mask (single channel)

    # Resize images to target size
    img = img.resize((target_w, target_h), Image.BILINEAR)
    target = target.resize((target_w, target_h), Image.NEAREST)  # For labels, use NEAREST to preserve discrete values

    # Convert to numpy arrays, pixel values 0-255
    img_np = np.array(img, dtype=np.float32)  # (H, W, 3)
    target_np = np.array(target, dtype=np.float32) / 255.0  # Normalize target to 0/1

    # Normalize image (which is 0-255 range) using the 0-1 range mean/std
    img_norm_np = img_np / 255.0  # Scale to 0-1
    img_norm_np = (img_norm_np - norm_mean) / norm_std

    # Convert to PyTorch tensor and permute to C, H, W, then add batch dim
    img_tensor = torch.from_numpy(img_norm_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    target_tensor = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return img_tensor, target_tensor, img, target  # Return original PIL images for display


# Load and preprocess the chosen image pair
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from xView2 test set...")
# Note: original_img and original_target are PIL images
input_image_tensor, gt_target_tensor, original_img, original_target = \
    load_and_preprocess_xview2_pair(XVIEW2_TEST_ROOT, IMAGE_PAIR_INDEX,
                                    TARGET_H, TARGET_W, NORM_MEAN, NORM_STD)

# Move input tensor to device
input_image_tensor = input_image_tensor.to(device)

print("Image pair loaded and preprocessed.")

# --- 3. Run Inference ---
print("Performing inference...")
with torch.no_grad():
    predictions = model(input_image_tensor)

# The model's primary output for semantic segmentation (classifier head) is expected to be [B, 1, H, W]
# Apply sigmoid and threshold
predicted_mask_prob = predictions.sigmoid()

# --- Ensure the predicted mask is 2D (H, W) for grayscale display ---
# Assuming channel 0 is the building mask probability.
predicted_mask = (predicted_mask_prob[:, 0, :, :] > 0.5).float()  # Select channel 0 and threshold
predicted_mask_np = predicted_mask.cpu().squeeze().numpy()  # Squeeze to remove batch dim if B=1

# Ground truth mask should also be 2D (H, W)
gt_target_np = gt_target_tensor.cpu().squeeze().numpy()

print("Inference complete.")

# --- 4. Visualize Results ---
print("Displaying results...")

plt.figure(figsize=(15, 5))  # Adjusted for 3 plots

plt.subplot(1, 3, 1)  # Changed to 1 row, 3 columns
plt.imshow(np.array(original_img))  # Display original PIL image converted to numpy
plt.title(f'Original Image (Index: {IMAGE_PAIR_INDEX})')
plt.axis('off')

plt.subplot(1, 3, 2)  # Changed to 1 row, 3 columns
plt.imshow(gt_target_np, cmap='gray')  # gt_target_np should be (H, W)
plt.title('Ground Truth Building Mask')
plt.axis('off')

plt.subplot(1, 3, 3)  # Changed to 1 row, 3 columns
plt.imshow(predicted_mask_np, cmap='gray')  # predicted_mask_np should be (H, W)
plt.title('Predicted Building Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nInference and visualization complete for the selected image pair.")