import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

source_folder = "dataset/Images/"
destination_folder = "dataset/Images_converted/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected {device} device")

if os.path.exists(destination_folder):
    user_input = input(
        "Dir 'Images_converted' already exists. Would you like to overwrite it anyway? (y/n): "
    )
    if user_input.lower() == "y":
        try:
            for root, dirs, files in os.walk(destination_folder):
                for file in files:
                    os.remove(os.path.join(root, file))
            print("Directory 'Images_converted' cleaned.")
        except OSError as e:
            print(f"Error cleaning directory: {e}")
            exit(1)
    else:
        print("Skipping directory cleaning and converting images to grayscale.")
        exit(1)
else:
    os.makedirs(destination_folder)
    print("Directory 'Images_converted' created.")


def convert_to_grayscale(images, destination_paths):
    try:
        img_tensors = [
            transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(device)
            for img in images
        ]
        img_batch = torch.cat(img_tensors, dim=0)

        grayscale_tensors = torch.einsum(
            "...chw,c->...hw",
            img_batch,
            torch.tensor([0.2989, 0.5870, 0.1140], device=device),
        )

        grayscale_images = [
            transforms.ToPILImage()(tensor.squeeze(0).cpu())
            for tensor in grayscale_tensors
        ]
        for img, path in zip(grayscale_images, destination_paths):
            img.save(path)
    except Exception as e:
        print(f"Error processing images: {e}")


image_paths = []
destination_paths = []
for subdir, _, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            source_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(source_path, source_folder)
            destination_path = os.path.join(destination_folder, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            destination_path = destination_path.replace("\\", "/")
            source_path = source_path.replace("\\", "/")
            image_paths.append(source_path)
            destination_paths.append(destination_path)

batch_size = 64  # allocated 12046MiB / 12288MiB
for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing", unit="batch"):
    batch_images = [
        Image.open(img_path).convert("RGB")
        for img_path in image_paths[i : i + batch_size]
    ]
    batch_dest_paths = destination_paths[i : i + batch_size]
    convert_to_grayscale(batch_images, batch_dest_paths)

print(f"Conversion to grayscale complete for {len(image_paths)} files.")
