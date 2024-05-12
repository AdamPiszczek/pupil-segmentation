import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.categories = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.image_paths = []
        self.mask_paths = glob.glob("dataset/Masks/*/*.png")

        for mask_path in self.mask_paths:
            corresponding_image_path = mask_path.replace(
                "Masks", "Images_converted"
            ).replace(".png", ".jpg")
            if os.path.exists(corresponding_image_path):
                self.image_paths.append(corresponding_image_path)

        # Check if each image has a corresponding mask
        for img_path in self.image_paths:
            corresponding_mask_path = img_path.replace(
                "Images_converted", "Masks"
            ).replace(".jpg", ".png")
            if corresponding_mask_path not in self.mask_paths:
                print(f"Missing mask for image: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        print(idx)
        print(len(self.image_paths))
        image = Image.open(self.image_paths[idx]).convert(
            "RGB"
        )  # Convert mask to grayscale
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Define transforms for images and masks
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize images and masks if necessary
        transforms.ToTensor(),
    ]
)

# Create a custom dataset
custom_dataset = CustomDataset(".", transform=transform)

# Create a DataLoader for batching and shuffling the data
data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)
