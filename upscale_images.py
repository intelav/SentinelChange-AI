# upscale_images.py
import os
import torch
from PIL import Image
from tqdm import tqdm
from RRDBNet_arch import RRDBNet
import numpy as np
import urllib.request
import sys
from collections import OrderedDict
import math


def download_model_if_needed(model_path, model_url):
    """
    Downloads the pre-trained model from the given URL if it doesn't already exist.
    """
    if os.path.exists(model_path):
        print(f"Model '{os.path.basename(model_path)}' already exists. Skipping download.")
        return

    print(f"Model not found at '{model_path}'. Downloading from URL...")

    # --- Progress Bar for Download ---
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            # Use carriage return to stay on the same line
            sys.stdout.write(f"\rDownloading model... {percent:.1f}% Complete")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(model_url, model_path, show_progress)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading the model: {e}")
        print("Please try downloading it manually and placing it in the correct directory.")
        # Clean up partially downloaded file if it exists
        if os.path.exists(model_path):
            os.remove(model_path)
        sys.exit(1)


def upscale_images(input_dir, output_dir, model_path='ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'):
    """
    Upscales all images in a directory using the ESRGAN model with tiling to conserve memory.
    """
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model Definition ---
    model = RRDBNet(3, 3, 64, 23, gc=32)
    try:
        load_net = torch.load(model_path)
        if 'params' in load_net:
            load_net = load_net['params']

        new_state_dict = OrderedDict()
        for k, v in load_net.items():
            name = k.replace('body.', 'RRDB_trunk.')
            name = name.replace('.rdb', '.RDB')
            name = name.replace('conv_body', 'trunk_conv')
            name = name.replace('conv_up1', 'upconv1')
            name = name.replace('conv_up2', 'upconv2')
            name = name.replace('conv_hr', 'HRconv')
            name = name.replace('conv_RRDB_trunk', 'trunk_conv')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    except FileNotFoundError:
        print(f"FATAL: Model file not found at '{model_path}'. The download may have failed.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"FATAL: Error loading model state_dict: {e}")
        sys.exit(1)

    model.eval()
    model = model.to(device)
    print(f"Model '{os.path.basename(model_path)}' loaded successfully.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for filename in tqdm(image_files, desc="Upscaling Images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # --- Tiled Inference Implementation ---
        try:
            img = Image.open(input_path).convert('RGB')

            # --- OPTIMIZATION: Increased tile size and added overlap ---
            tile_size = 768  # Size of the tiles to process
            tile_pad = 10  # Overlap between tiles to avoid artifacts
            scale = 4  # The upscale factor of the model

            # Create a new blank image to paste the upscaled tiles into
            final_output_image = Image.new('RGB', (img.width * scale, img.height * scale))

            # Calculate the number of tiles
            num_tiles_w = math.ceil(img.width / tile_size)
            num_tiles_h = math.ceil(img.height / tile_size)

            # Loop through each tile
            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    # Define the crop box for the current tile with padding
                    left = j * tile_size
                    top = i * tile_size
                    right = min((j + 1) * tile_size, img.width)
                    bottom = min((i + 1) * tile_size, img.height)

                    # Add padding
                    left_pad = max(left - tile_pad, 0)
                    top_pad = max(top - tile_pad, 0)
                    right_pad = min(right + tile_pad, img.width)
                    bottom_pad = min(bottom + tile_pad, img.height)

                    # Crop the padded tile from the input image
                    tile_img = img.crop((left_pad, top_pad, right_pad, bottom_pad))

                    # Convert the tile to a tensor
                    tile_np = np.array(tile_img) * 1.0 / 255
                    tile_tensor = torch.from_numpy(np.transpose(tile_np[:, :, [2, 1, 0]], (2, 0, 1))).float()
                    tile_lr = tile_tensor.unsqueeze(0).to(device)

                    # --- OPTIMIZATION: Use mixed precision (autocast) ---
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output_tile = model(tile_lr)

                    output_tile = output_tile.data.squeeze().float().cpu().clamp_(0, 1).numpy()

                    # Convert the output tensor back to an image
                    output_tile = np.transpose(output_tile[[2, 1, 0], :, :], (1, 2, 0))
                    output_tile_img = Image.fromarray((output_tile * 255.0).round().astype(np.uint8))

                    # --- Remove the padded areas from the upscaled tile ---
                    pad_left_scaled = (left - left_pad) * scale
                    pad_top_scaled = (top - top_pad) * scale
                    pad_right_scaled = (right_pad - right) * scale
                    pad_bottom_scaled = (bottom_pad - bottom) * scale

                    crop_out_box = (
                        pad_left_scaled,
                        pad_top_scaled,
                        output_tile_img.width - pad_right_scaled,
                        output_tile_img.height - pad_bottom_scaled
                    )
                    output_tile_img = output_tile_img.crop(crop_out_box)

                    # Paste the upscaled tile into the final output image
                    paste_x = j * tile_size * scale
                    paste_y = i * tile_size * scale
                    final_output_image.paste(output_tile_img, (paste_x, paste_y))

            # Save the final stitched image
            final_output_image.save(output_path)

        except Exception as e:
            print(f"\nAn error occurred while processing {filename}: {e}")
            # Optionally, skip to the next image
            continue


if __name__ == '__main__':
    # --- Configuration ---
    MODEL_FILENAME = 'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
    MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'

    download_model_if_needed(MODEL_FILENAME, MODEL_URL)

    print("\n--- Upscaling images in directory 'A' ---")
    upscale_images('MyCustomCD_Dataset/train/A', 'MyCustomCD_Dataset_Upscaled/train/A', MODEL_FILENAME)

    print("\n--- Upscaling images in directory 'B' ---")
    upscale_images('MyCustomCD_Dataset/train/B', 'MyCustomCD_Dataset_Upscaled/train/B', MODEL_FILENAME)

    print("\n--- All images have been upscaled successfully! ---")
