import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys

# Add the ChangeStar project root to the Python path
# This assumes you are running this script from the ChangeStar project's root directory
sys.path.insert(0, os.getcwd())

import ever as er
from ever.core import config as ever_config
from ever.core.checkpoint import CheckPoint, remove_module_prefix

er.registry.register_all()

# --- Configuration for Inference ---
# IMPORTANT: Update these paths to match your SYSU-CD setup.

# 1. Path to the model checkpoint file trained on SYSU-CD
CHECKPOINT_PATH = 'log/finetune-SYSUCD/r50_farseg_changestar/model-9750.pth' # <-- MUST-CHANGE: Path to your SYSU-CD checkpoint
#CHECKPOINT_PATH = 'log/bisup-LEVIRCD/r50_farseg_changestar/model-5600.pth'
# 2. Root directory of your SYSU-CD dataset
SYSU_CD_ROOT = '/media/avaish/aiwork/satellite-work/datasets/sysu-cd'  # <-- MUST-CHANGE: Path to your SYSU-CD dataset root

# 3. Path to the configuration file used for training on SYSU-CD
CONFIG_PATH = 'configs/sysucd/r50_farseg_changestar_finetune.py' # <-- MUST-CHANGE: Path to your SYSU-CD config file

# Index of the image pair you want to use from the validation set (0 to 3999 for SYSU-CD)
IMAGE_PAIR_INDEX = 3000 # You can change this to test different image pairs.good image pairs : 18,19,25,26,3599,3299,3000,2000
# -----------------------------------

print(f"Loading configuration from: {CONFIG_PATH}")
if not os.path.exists(CONFIG_PATH):
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    print("Please ensure the CONFIG_PATH variable is set correctly.")
    sys.exit(1)
cfg = ever_config.import_config(CONFIG_PATH)

# --- 1. Model Construction ---
print("Constructing model...")
model = er.builder.make_model(cfg.model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
    print("Please ensure the CHECKPOINT_PATH variable is set correctly.")
    sys.exit(1)

ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

# Adjust state_dict keys if the model was saved with DataParallel/DistributedDataParallel
model_state_dict = ckpt.get(CheckPoint.MODEL)
if model_state_dict:
    model.load_state_dict(remove_module_prefix(model_state_dict))
else:
    # Fallback for checkpoints that might not use the CheckPoint.MODEL key
    model.load_state_dict(remove_module_prefix(ckpt))

model.eval()  # Set model to evaluation mode

print("Model loaded successfully.")


# --- 2. Load and Preprocess a Single Image Pair ---
def load_and_preprocess_image_pair(root_dir, split, index, target_size=(256, 256)):
    """
    Loads and preprocesses an image pair from the SYSU-CD dataset.
    """
    split_dir = os.path.join(root_dir, split)
    # Corrected directory names for SYSU-CD: time1, time2
    image_t1_files = sorted(os.listdir(os.path.join(split_dir, 'time1')))
    image_t2_files = sorted(os.listdir(os.path.join(split_dir, 'time2')))
    label_files = sorted(os.listdir(os.path.join(split_dir, 'label')))

    img_t1_path = os.path.join(split_dir, 'time1', image_t1_files[index])
    img_t2_path = os.path.join(split_dir, 'time2', image_t2_files[index])
    label_path = os.path.join(split_dir, 'label', label_files[index])

    img_t1 = Image.open(img_t1_path).convert('RGB')
    img_t2 = Image.open(img_t2_path).convert('RGB')
    label = Image.open(label_path).convert('L')  # Ground truth change map

    # --- Preprocessing ---
    # Resize images to the target size (SYSU-CD images are typically 256x256)
    img_t1_resized = img_t1.resize(target_size, Image.BILINEAR)
    img_t2_resized = img_t2.resize(target_size, Image.BILINEAR)

    # Convert to numpy arrays
    img_t1_np = np.array(img_t1_resized, dtype=np.float32)
    img_t2_np = np.array(img_t2_resized, dtype=np.float32)
    label_np = np.array(label.resize(target_size, Image.NEAREST), dtype=np.float32) / 255.0

    # Apply ImageNet normalization statistics (as is common)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    img_t1_norm = (img_t1_np - mean) / std
    img_t2_norm = (img_t2_np - mean) / std

    # Convert to PyTorch tensors and permute to (C, H, W)
    img_t1_tensor = torch.from_numpy(img_t1_norm).permute(2, 0, 1)
    img_t2_tensor = torch.from_numpy(img_t2_norm).permute(2, 0, 1)
    label_tensor = torch.from_numpy(label_np).unsqueeze(0)  # Add channel dim

    # Concatenate time1 and time2 images along the channel dimension
    # Model expects [B, 6, H, W]
    bi_images_tensor = torch.cat([img_t1_tensor, img_t2_tensor], dim=0).unsqueeze(0)  # Add batch dim

    # Return original (non-resized) images for clearer visualization
    return bi_images_tensor, label_tensor.unsqueeze(0), img_t1, img_t2, label


# Load and preprocess the chosen image pair from the 'val' split
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from SYSU-CD 'val' set...")
bi_images_tensor, gt_label_tensor, original_img_t1, original_img_t2, original_label = \
    load_and_preprocess_image_pair(SYSU_CD_ROOT, 'val', IMAGE_PAIR_INDEX)

# Move tensors to the selected device
bi_images_tensor = bi_images_tensor.to(device)

print("Image pair loaded and preprocessed.")

# --- 3. Run Inference ---
print("Performing inference...")
with torch.no_grad():
    predictions = model(bi_images_tensor)

# The model output is expected to be logits for the change map.
# Apply sigmoid to get probabilities and threshold to get the binary map.
predicted_change_prob = predictions.sigmoid()
predicted_change_map = (predicted_change_prob > 0.5).float()

# Move prediction to CPU and convert to numpy for visualization
predicted_change_map_np = predicted_change_map.cpu().squeeze().numpy()
gt_label_np = gt_label_tensor.cpu().squeeze().numpy()

print("Inference complete.")

# --- 4. Visualize Results ---
print("Displaying results...")

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(original_img_t1)
plt.title(f'Time 1 (Index: {IMAGE_PAIR_INDEX})')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(original_img_t2)
plt.title('Time 2')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(original_label, cmap='gray')
plt.title('Ground Truth Change')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(predicted_change_map_np, cmap='gray')
plt.title('Predicted Change')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\nInference and visualization complete for image pair {IMAGE_PAIR_INDEX}.")