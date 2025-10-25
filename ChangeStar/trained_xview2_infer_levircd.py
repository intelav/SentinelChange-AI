import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys

# Add the ChangeStar project root to the Python path
# This assumes you are running the Jupyter notebook from inside /media/avaish/aiwork/satellite-work/work-changestar/ChangeStar/
sys.path.insert(0, os.getcwd())

import ever as er
from ever.core import config as ever_config
from ever.core import to
from ever.core.checkpoint import CheckPoint, remove_module_prefix # Ensure remove_module_prefix is imported

er.registry.register_all() # Crucial for model building

# --- Configuration for Inference ---
# Path to your specific checkpoint file (trained on xView2)
CHECKPOINT_PATH = 'log/changestar_sisup/backup_xview2_training_levircd_eval/model-40000.pth'

# Root directory of your LEVIR-CD dataset for validation data
LEVIR_CD_VAL_ROOT = '/media/avaish/aiwork/satellite-work/datasets/LEVIR-CD'

# Index of the image pair you want to use from the validation set (0 to 64)
IMAGE_PAIR_INDEX = 30

# --- Model Configuration (should match your training config) ---
# This is loaded from your config file
CONFIG_PATH = 'configs/trainxView2/r50_farseg_changemixin_symmetry.py'

# --- Image Preprocessing Parameters ---
# These mean/std values are from standard.py's LEVIR-CD test set normalization (6-channel input)
# Note: xView2 training `standard.py` has 3-channel normalization for its `train` part,
# but the model itself (ChangeStar R50 with detector) expects 6-channel bi-temporal inputs for change detection.
# So, for inference on LEVIR-CD, we use 6-channel normalization.
NORM_MEAN = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32)

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
# Adjust state_dict keys if the model was saved with DataParallel/DistributedDataParallel
if hasattr(model, 'module'): # If it's a DDP model, but we're loading onto a single device
    model.module.load_state_dict(remove_module_prefix(model_state_dict))
else: # If it's a single-GPU model, load directly
    model.load_state_dict(remove_module_prefix(model_state_dict))

model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# --- 2. Load and Preprocess a Single Image Pair ---
def load_and_preprocess_levircd_pair(root_dir, split, index, target_h, target_w, norm_mean, norm_std):
    split_dir = os.path.join(root_dir, split)
    image_a_files = sorted(os.listdir(os.path.join(split_dir, 'A')))
    image_b_files = sorted(os.listdir(os.path.join(split_dir, 'B')))
    label_files = sorted(os.listdir(os.path.join(split_dir, 'label')))

    if index >= len(image_a_files):
        raise IndexError(f"Index {index} out of bounds for split '{split}'. Max index is {len(image_a_files) - 1}.")

    img_a_path = os.path.join(split_dir, 'A', image_a_files[index])
    img_b_path = os.path.join(split_dir, 'B', image_b_files[index])
    label_path = os.path.join(split_dir, 'label', label_files[index])

    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')
    label = Image.open(label_path).convert('L') # Ground truth change map (single channel)

    # Resize images to target size
    img_a = img_a.resize((target_w, target_h), Image.BILINEAR)
    img_b = img_b.resize((target_w, target_h), Image.BILINEAR)
    label = label.resize((target_w, target_h), Image.NEAREST) # For labels, use NEAREST to preserve discrete values

    # Convert to numpy arrays, pixel values 0-255
    img_a_np = np.array(img_a, dtype=np.float32) # (H, W, 3)
    img_b_np = np.array(img_b, dtype=np.float32) # (H, W, 3)
    label_np = np.array(label, dtype=np.float32) / 255.0 # Normalize label to 0/1

    # Concatenate T1 and T2 images: (H, W, 6)
    bi_images_np = np.concatenate([img_a_np, img_b_np], axis=-1)

    # Normalize bi_images_np (which is 0-255 range) using the 0-1 range mean/std
    # This requires converting to 0-1 range first, then applying normalization as if albumentations would.
    bi_images_norm_np = bi_images_np / 255.0
    bi_images_norm_np = (bi_images_norm_np - norm_mean) / norm_std

    # Convert to PyTorch tensor and permute to C, H, W
    bi_images_tensor = torch.from_numpy(bi_images_norm_np).permute(2, 0, 1).unsqueeze(0) # (1, 6, H, W)
    label_tensor = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    return bi_images_tensor, label_tensor, img_a, img_b, label # Return original PIL images for display

# Load and preprocess the chosen image pair
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from LEVIR-CD val set...")
bi_images_tensor, gt_label_tensor, original_img_a, original_img_b, original_label = \
    load_and_preprocess_levircd_pair(LEVIR_CD_VAL_ROOT, 'val', IMAGE_PAIR_INDEX,
                                      TARGET_H, TARGET_W, NORM_MEAN, NORM_STD)

# Move input tensor to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bi_images_tensor = bi_images_tensor.to(device)

print("Image pair loaded and preprocessed.")

# --- 3. Run Inference ---
print("Performing inference...")
with torch.no_grad():
    predictions = model(bi_images_tensor) # model(bi_images_tensor) should return the change map tensor

# The model's primary output (for change detection) is expected to be [B, 1, H, W]
# Apply sigmoid and threshold
predicted_change_prob = predictions.sigmoid()

# --- CRITICAL FIX: Ensure the predicted change map is 2D (H, W) for grayscale imshow ---
# If predictions were [B, C, H, W] and C>1 (e.g., C=3), you need to select the correct channel.
# Assuming channel 0 is the change probability.
predicted_change_map = (predicted_change_prob[:, 0, :, :] > 0.5).float() # Select channel 0 and threshold
predicted_change_map_np = predicted_change_map.cpu().squeeze().numpy() # Squeeze to remove batch dim if B=1

# Ensure ground truth is also 2D (H, W)
gt_label_np = gt_label_tensor.cpu().squeeze().numpy()

print("Inference complete.")

# --- 4. Visualize Results ---
print("Displaying results...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(np.array(original_img_a)) # Display original PIL images converted to numpy
plt.title(f'Time 1 (Index: {IMAGE_PAIR_INDEX})')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.array(original_img_b)) # Display original PIL images converted to numpy
plt.title('Time 2')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gt_label_np, cmap='gray') # gt_label_np should be (H, W)
plt.title('Ground Truth Change')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(predicted_change_map_np, cmap='gray') # predicted_change_map_np should be (H, W)
plt.title('Predicted Change')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nInference and visualization complete for the selected image pair.")