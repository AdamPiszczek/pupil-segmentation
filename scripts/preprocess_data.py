import os

from PIL import Image
from tqdm import tqdm

source_folder = "Images/"
destination_folder = "Images_converted/"

if os.path.exists("Images_converted/"):
    user_input = input(
        "Dir 'Images_converted' already exists. Would you like to overwrite it anyway? (y/n): "
    )
    if user_input.lower() == "y":
        try:
            os.rmdir(destination_folder)
            os.makedirs(destination_folder)
            print("Directory 'Images_converted' created.")
        except OSError as e:
            print(f"Error creating directory: {e}")
    else:
        print("Skipping directory cleaning and converting images to grayscale.")
        exit(1)
else:
    os.makedirs("Images_converted")


def convert_to_grayscale(source_path, destination_path):
    try:
        img = Image.open(source_path).convert("L")
        img.save(destination_path)
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        exit(1)


subdirectories = [
    os.path.join(destination_folder, d)
    for d in os.listdir(destination_folder)
    if os.path.isdir(os.path.join(destination_folder, d))
]
for subdir in subdirectories:
    for _, _, files in os.walk(subdir):
        for file in files:
            os.remove(os.path.join(subdir, file))
    os.rmdir(subdir)

num_of_files = 0
for subdir, _, files in os.walk(source_folder):
    num_of_files = len(files)
    for file in tqdm(files, desc="Files", unit="file", leave=False):
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            source_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(source_path, source_folder)
            destination_path = os.path.join(destination_folder, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            destination_path = destination_path.replace("\\", "/")
            source_path = source_path.replace("\\", "/")
            convert_to_grayscale(source_path, destination_path)

print(f"Conversion to grayscale complete! for {num_of_files} files.")
