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
from ever.core.checkpoint import CheckPoint, remove_module_prefix

er.registry.register_all()

# --- Configuration for Inference ---
# Path to your specific checkpoint file
#CHECKPOINT_PATH = './checkpoints/model-5600.pth'
CHECKPOINT_PATH = 'log/bisup-LEVIRCD/r50_farseg_changestar/model-5600.pth'
#CHECKPOINT_PATH = 'log/finetune-SYSUCD/r50_farseg_changestar/model-2000.pth'
# Root directory of your LEVIR-CD dataset
LEVIR_CD_VAL_ROOT = '/media/avaish/aiwork/satellite-work/datasets/LEVIR-CD'

# Index of the image pair you want to use from the validation set (0 to 64)
# You can change this to test different image pairs.
IMAGE_PAIR_INDEX = 27

# --- Model Configuration (should match your training config) ---
# This is loaded from your config file
# Ensure this path is correct relative to your ChangeStar project root
#CONFIG_PATH = 'configs/levircd/r18_farseg_changestar_bisup.py'
CONFIG_PATH = 'configs/levircd/r50_farseg_changestar_bisup.py'
# -----------------------------------

print(f"Loading configuration from: {CONFIG_PATH}")
cfg = ever_config.import_config(CONFIG_PATH)

# --- 1. Model Construction ---
# Use the same model builder as during training
model = er.builder.make_model(cfg.model)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# If the model was trained with DistributedDataParallel (DDP), remove 'module.' prefix
# This is typically handled by `ever.core.checkpoint.remove_module_prefix`
# but we'll manually load the state dict here.
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Adjust state_dict keys if the model was saved with DataParallel/DistributedDataParallel
# The CheckPoint.load function does this automatically for `model_state_dict`
# We manually extract it here for clarity.
model_state_dict = ckpt[CheckPoint.MODEL]
if hasattr(model, 'module'): # If it's a DDP model, but we're loading onto a single device
    model.module.load_state_dict(ever_config.remove_module_prefix(model_state_dict))
else: # If it's a single-GPU model, just load directly
    model.load_state_dict(remove_module_prefix(model_state_dict)) # Use the helper function to handle DDP prefix if present

model.eval() # Set model to evaluation mode

print("Model loaded successfully.")

# --- 2. Load and Preprocess a Single Image Pair ---
def load_and_preprocess_image_pair(root_dir, split, index, transforms):
    split_dir = os.path.join(root_dir, split)
    image_a_files = sorted(os.listdir(os.path.join(split_dir, 'A')))
    image_b_files = sorted(os.listdir(os.path.join(split_dir, 'B')))
    label_files = sorted(os.listdir(os.path.join(split_dir, 'label'))) # Assuming label exists for val

    img_a_path = os.path.join(split_dir, 'A', image_a_files[index])
    img_b_path = os.path.join(split_dir, 'B', image_b_files[index])
    label_path = os.path.join(split_dir, 'label', label_files[index])

    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')
    label = Image.open(label_path).convert('L') # Ground truth change map (single channel)

    # Apply transforms (e.g., normalization, ToTensor)
    # This part can be tricky as `ever`'s transform pipeline is specific.
    # We'll try to mimic the essential ones.
    # For LEVIR-CD, images are often 1024x1024, but the model might expect 512x512
    # or it might handle different sizes if input_stride allows.
    # Given your training config's `input_size` (if defined), you'd usually resize here.
    # For simplicity, we'll assume the model handles the input size, or apply common transforms.

    # Example: Simple ToTensor and Concatenation.
    # You might need to replicate the exact `ever` data transforms from `standard.data.train.params.transforms`
    # and `standard.data.test.params.transforms` here for optimal results.
    # These often include normalization and resizing.
    # For now, let's use a basic manual transform.

    # Convert to numpy arrays first for easier manipulation
    img_a_np = np.array(img_a, dtype=np.float32)
    img_b_np = np.array(img_b, dtype=np.float32)
    label_np = np.array(label, dtype=np.float32) / 255.0 # Normalize label to 0/1

    # Apply normalization (example from your config for ImageNet stats)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    img_a_norm = (img_a_np - mean) / std
    img_b_norm = (img_b_np - mean) / std

    # Convert to PyTorch tensors and permute to C, H, W
    img_a_tensor = torch.from_numpy(img_a_norm).permute(2, 0, 1)
    img_b_tensor = torch.from_numpy(img_b_norm).permute(2, 0, 1)
    label_tensor = torch.from_numpy(label_np).unsqueeze(0) # Add channel dim for label

    # Concatenate time1 and time2 images along the channel dimension
    # Your model expects [b, tc, h, w], where tc=6 (3 channels for t1 + 3 channels for t2)
    bi_images_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=0).unsqueeze(0) # Add batch dim

    return bi_images_tensor, label_tensor.unsqueeze(0), img_a, img_b, label # Also return original PIL images for display

# Load and preprocess the chosen image pair
print(f"Loading image pair index {IMAGE_PAIR_INDEX} from LEVIR-CD val set...")
bi_images_tensor, gt_label_tensor, original_img_a, original_img_b, original_label = \
    load_and_preprocess_image_pair(LEVIR_CD_VAL_ROOT, 'val', IMAGE_PAIR_INDEX, None) # Transforms handled manually

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bi_images_tensor = bi_images_tensor.to(device)

print("Image pair loaded and preprocessed.")

# --- 3. Run Inference ---
print("Performing inference...")
with torch.no_grad():
    predictions = model(bi_images_tensor)

# The output format for ChangeStarBiSup is crucial.
# Based on r18_farseg_changestar_bisup.py and train_sup_change.py,
# the model output directly represents the change map probability.
# It's likely a tensor of shape [B, 1, H, W]
# Apply sigmoid and threshold
predicted_change_prob = predictions.sigmoid()
predicted_change_map = (predicted_change_prob > 0.5).float()

# Move prediction to CPU and convert to numpy
predicted_change_map_np = predicted_change_map.cpu().squeeze().numpy()
gt_label_np = gt_label_tensor.cpu().squeeze().numpy() # Squeeze to remove batch/channel dims

print("Inference complete.")

# --- 4. Visualize Results ---
print("Displaying results...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(original_img_a)
plt.title(f'Time 1 (Index: {IMAGE_PAIR_INDEX})')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(original_img_b)
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