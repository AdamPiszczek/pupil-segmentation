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
            corresponding_image_path = mask_path.replace("Masks", "Images").replace(
                ".png", ".jpg"
            )
            if os.path.exists(corresponding_image_path):
                self.image_paths.append(corresponding_image_path)

        # Check if each image has a corresponding mask
        for img_path in self.image_paths:
            corresponding_mask_path = img_path.replace("Images", "Masks").replace(
                ".jpg", ".png"
            )
            if corresponding_mask_path not in self.mask_paths:
                print(f"Missing mask for image: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if os.path.exists(self.image_paths[idx]):
            image = Image.open(self.image_paths[idx]).convert("RGB")
        else:
            print(f"Path '{self.image_paths[idx]}' does not exists!")

        if os.path.exists(self.mask_paths[idx]):
            mask = Image.open(self.mask_paths[idx]).convert("RGB")
        else:
            print(f"Path '{self.mask_paths[idx]}' does not exists!")

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
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)


# Define a simple semantic segmentation model
class SimpleSegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Found device: '{device}'")
model = SimpleSegmentationModel().to(device)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print(f"Starting training SimpleSegmentationModel for {epochs} epochs")
for epoch in range(5):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, masks in data_loader:
        optimizer.zero_grad()

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # Flatten the output and masks for the loss function
        loss = criterion(outputs.squeeze(), masks.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(custom_dataset)
    print(f"Epoch [{epoch + 1}/5], Loss: {epoch_loss}")


# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_score = 0.0
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # For demonstration purposes, calculating a simple score (average loss)
            loss = criterion(outputs.squeeze(), masks.squeeze())
            total_score += loss.item() * images.size(0)

        average_score = total_score / len(custom_dataset)
        return average_score


# Evaluate the model on the training dataset
train_score = evaluate_model(model, data_loader)
print(f"Training Score: {train_score}")
