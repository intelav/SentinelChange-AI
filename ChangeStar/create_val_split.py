# File Location: ChangeStar/create_val_split.py

import os
import glob
import random
import shutil
from tqdm import tqdm


def create_validation_split(dataset_dir, val_split_ratio=0.2):
    """
    Creates a validation set by moving a percentage of files from the train directory.
    """
    print(f"Creating validation split for dataset at: {dataset_dir}")

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    # Define source and destination directories
    train_a = os.path.join(train_dir, 'A')
    train_b = os.path.join(train_dir, 'B')
    train_label = os.path.join(train_dir, 'label')

    val_a = os.path.join(val_dir, 'A')
    val_b = os.path.join(val_dir, 'B')
    val_label = os.path.join(val_dir, 'label')

    # Create validation directories if they don't exist
    os.makedirs(val_a, exist_ok=True)
    os.makedirs(val_b, exist_ok=True)
    os.makedirs(val_label, exist_ok=True)
    print("Validation directories created.")

    # Get a list of all training images from directory 'A'
    all_train_files = glob.glob(os.path.join(train_a, '*.png'))
    if not all_train_files:
        print("Error: No training files found in 'train/A'. Please check the path.")
        return

    # Shuffle the files randomly
    random.shuffle(all_train_files)

    # Determine the number of files to move
    num_to_move = int(len(all_train_files) * val_split_ratio)
    files_to_move = all_train_files[:num_to_move]

    print(f"Total training files: {len(all_train_files)}")
    print(f"Moving {num_to_move} files to the validation set...")

    # Move the corresponding files for A, B, and label
    for file_path_a in tqdm(files_to_move, desc="Moving files"):
        basename = os.path.basename(file_path_a)

        # Define source paths for B and label
        file_path_b = os.path.join(train_b, basename)
        file_path_label = os.path.join(train_label, basename)

        # Define destination paths
        dest_path_a = os.path.join(val_a, basename)
        dest_path_b = os.path.join(val_b, basename)
        dest_path_label = os.path.join(val_label, basename)

        # Move the files
        if os.path.exists(file_path_a):
            shutil.move(file_path_a, dest_path_a)
        if os.path.exists(file_path_b):
            shutil.move(file_path_b, dest_path_b)
        if os.path.exists(file_path_label):
            shutil.move(file_path_label, dest_path_label)

    print("\nValidation split created successfully.")
    print(f"New training set size: {len(glob.glob(os.path.join(train_a, '*.png')))}")
    print(f"New validation set size: {len(glob.glob(os.path.join(val_a, '*.png')))}")


if __name__ == '__main__':
    # The path should be relative to where you run the script (the ChangeStar directory)
    DATASET_DIRECTORY = '../data/MyCustomCD_Dataset'
    create_validation_split(DATASET_DIRECTORY)

