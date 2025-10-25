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
from ever.core.checkpoint import CheckPoint, remove_module_prefix # Import remove_module_prefix

# Ensure all ever registry components are registered
er.registry.register_all()

# --- Configuration for Inference ---
# Path to your specific checkpoint file (trained on LEVIR-CD)
CHECKPOINT_PATH = './checkpoints/model-5600.pth' # Use your final LEVIR-CD trained checkpoint

# Root directory of your S2Looking dataset
S2LOOKING_ROOT = '/media/avaish/aiwork/satellite-work/datasets/S2Looking'

# Which split of S2Looking to use (e.g., 'val', 'test', 'train')
DATA_SPLIT = 'val'

# Index of the image pair you want to use from the selected split
# You can change this to test different image pairs.
IMAGE_PAIR_INDEX = 0 # S2Looking images are 1024x1024. Your model was trained on 512x512 patches.
                     # This means you'll need to either resize the S2Looking image or
                     # process it in 512x512 patches. For this example, we'll resize.

# --- Model Configuration (should ideally match your training config) ---
# If you have a specific config for S2Looking, use that.
# Otherwise, use the LEVIR-CD one as the model architecture is the same.
CONFIG_PATH = 'configs/levircd/r18_farseg_changestar_bisup.py'


# --- Image Preprocessing Parameters ---
# These should match your training's Normalize transform in standard.py
# For LEVIR-CD, you used mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406) and std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
# This implies two sets of ImageNet means/stds for the 6 channels (T1_RGB + T2_RGB)
# For S2Looking, which is multi-spectral, these means/stds might not be perfectly appropriate.
# However, for demonstration, we will use them as they were used in training.
NORM_MEAN = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32) * 255
NORM_STD = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32) * 255


# Desired input size for the model (your model was trained on 512x512 patches)
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
if hasattr(model, 'module'): # If it was wrapped in DDP
    model.module.load_state_dict(remove_module_prefix(model_state_dict))
else:
    model.load_state_dict(remove_module_prefix(model_state_dict))

model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# --- 2. Load and Preprocess a Single S2Looking Image Pair ---
def load_and_preprocess_s2looking_pair(root_dir, split, index, target_h, target_w, norm_mean, norm_std):
    split_dir = os.path.join(root_dir, split)

    # S2Looking directory structure: Image1, Image2, label (for original change GT)
    # Check if files exist for the given index
    image1_files = sorted(os.listdir(os.path.join(split_dir, 'Image1')))
    image2_files = sorted(os.listdir(os.path.join(split_dir, 'Image2')))
    label_files = sorted(os.listdir(os.path.join(split_dir, 'label')))

    if index >= len(image1_files):
        raise IndexError(f"Index {index} out of bounds for split '{split}'. Max index is {len(image1_files) - 1}.")

    img1_path = os.path.join(split_dir, 'Image1', image1_files[index])
    img2_path = os.path.join(split_dir, 'Image2', image2_files[index])
    label_path = os.path.join(split_dir, 'label', label_files[index]) # Original ground truth

    # Load images as RGB (assuming ChangeStar expects 3 channels per image)
    # S2Looking images are multi-spectral, but for ChangeStar, you might be using only RGB bands
    # or the first 3 bands. If your LEVIR-CD training used 3-channel images, stick to that.
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    label = Image.open(label_path).convert('L') # Ground truth change map (single channel)

    # Resize images to TARGET_H, TARGET_W (e.g., 512x512)
    img1 = img1.resize((target_w, target_h), Image.BILINEAR)
    img2 = img2.resize((target_w, target_h), Image.BILINEAR)
    label = label.resize((target_w, target_h), Image.NEAREST) # For labels, use NEAREST to preserve discrete values

    # Convert to numpy arrays
    img1_np = np.array(img1, dtype=np.float32) # Shape: (H, W, 3)
    img2_np = np.array(img2, dtype=np.float32) # Shape: (H, W, 3)
    label_np = np.array(label, dtype=np.float32) / 255.0 # Normalize label to 0/1

    # Apply normalization (mean/std for 6 channels)
    # Reshape mean/std for broadcasting: (1, 1, C)
    norm_mean_t1 = norm_mean[:3].reshape(1, 1, 3)
    norm_std_t1 = norm_std[:3].reshape(1, 1, 3)
    norm_mean_t2 = norm_mean[3:].reshape(1, 1, 3)
    norm_std_t2 = norm_std[3:].reshape(1, 1, 3)

    img1_norm = (img1_np - norm_mean_t1) / norm_std_t1
    img2_norm = (img2_np - norm_mean_t2) / norm_std_t2

    # Convert to PyTorch tensors and permute to C, H, W
    img1_tensor = torch.from_numpy(img1_norm).permute(2, 0, 1) # (3, H, W)
    img2_tensor = torch.from_numpy(img2_norm).permute(2, 0, 1) # (3, H, W)
    label_tensor = torch.from_numpy(label_np).unsqueeze(0) # (1, H, W)

    # Concatenate time1 and time2 images along the channel dimension
    # Model expects [b, tc, h, w], where tc=6 (3 channels for t1 + 3 channels for t2)
    bi_images_tensor = torch.cat([img1_tensor, img2_tensor], dim=0).unsqueeze(0) # (1, 6, H, W)

    return bi_images_tensor, label_tensor.unsqueeze(0), img1, img2, label # Also return original PIL images for display

# Load and preprocess the chosen image pair
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from S2Looking '{DATA_SPLIT}' set...")
bi_images_tensor, gt_label_tensor, original_img1, original_img2, original_label = \
    load_and_preprocess_s2looking_pair(S2LOOKING_ROOT, DATA_SPLIT, IMAGE_PAIR_INDEX,
                                      TARGET_H, TARGET_W, NORM_MEAN, NORM_STD)

# Move to device
bi_images_tensor = bi_images_tensor.to(device)

print("Image pair loaded and preprocessed.")

# --- 3. Run Inference ---
print("Performing inference...")
with torch.no_grad():
    predictions = model(bi_images_tensor)

# The output format for ChangeStarBiSup: a single-channel probability map [B, 1, H, W]
predicted_change_prob = predictions.sigmoid()
predicted_change_map = (predicted_change_prob > 0.5).float()

# Move prediction to CPU and convert to numpy
predicted_change_map_np = predicted_change_map.cpu().squeeze().numpy()
gt_label_np = gt_label_tensor.cpu().squeeze().numpy() # Squeeze to remove batch/channel dims

print("Inference complete.")

# --- 4. Visualize Results ---
print("Displaying results...")

plt.figure(figsize=(18, 6)) # Adjust figure size for better viewing

plt.subplot(1, 4, 1)
plt.imshow(original_img1)
plt.title(f'Time 1 ({DATA_SPLIT} Index: {IMAGE_PAIR_INDEX})')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(original_img2)
plt.title('Time 2')
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

print("\nInference and visualization complete for the selected image pair.")