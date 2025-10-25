import os
import zipfile
import fnmatch
from tqdm import tqdm

def find_files(directory, pattern):
    """Recursively finds all files matching the pattern."""
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def unzip_all(directory):
    """Unzips all .SAFE.zip files in the specified directory."""
    zip_files = list(find_files(directory, '*.SAFE.zip'))
    if not zip_files:
        print("No .SAFE.zip files found to unzip.")
        return

    print(f"Found {len(zip_files)} .SAFE.zip files to extract.")
    for zip_path in tqdm(zip_files, desc="Unzipping files"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a directory with the same name as the zip file (without .zip)
                extract_path = zip_path.rsplit('.zip', 1)[0]
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)
                zip_ref.extractall(extract_path)
            print(f"Successfully extracted {os.path.basename(zip_path)}")
            # Optional: Remove the zip file after extraction to save space
            # os.remove(zip_path)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        except Exception as e:
            print(f"An error occurred while unzipping {zip_path}: {e}")

if __name__ == '__main__':
    # The root directory containing your terrain folders (snow-2023, desert-2022, etc.)
    DATA_ROOT = './'
    unzip_all(DATA_ROOT)
    print("\nAll files have been unzipped.")
