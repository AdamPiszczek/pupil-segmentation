import datetime
import os
from itertools import product
from typing import Any

# import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
import transforms as T
from engine import evaluate, evaluate_loss, train_one_epoch
from PIL import Image
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNNPredictor,
)
from torchvision.transforms import Compose

import utils


class PupilDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None) -> None:
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "converted_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "converted_masks"))))

    def __getitem__(self, idx) -> tuple:
        img_path = os.path.join(self.root, "converted_images", self.imgs[idx])
        mask_path = os.path.join(self.root, "converted_masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            # testing correctness of the bounding boxes
            # img = cv2.imread(img_path)
            # pt1 = (xmin, ymin)  # Top-left corner
            # pt2 = (xmax, ymax)
            # cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)
            # text_position = (pt1[0], pt1[1] - 10)
            # cv2.putText(
            #     img,
            #     "pupi;",
            #     text_position,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     2,
            #     (0, 255, 0),
            #     thickness=2,
            # )
            # cv2.imwrite(f"results/test_{i}_{idx}.png", img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)


def build_model(num_classes: Any) -> MaskRCNN:
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # model = torch.load("models/model_20240120-161444.pt")

    return model


def get_transform(train: bool) -> Compose:
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == "__main__":
    learning_rates = [0.01]
    momentums = [0.9]
    batch_sizes = [
        8
    ]  # 13.78 GiB is allocated by PyTorch, and 475.22 MiB is reserved by PyTorch but unallocated. I have 8 GB
    num_epochs_list = [100]
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for lr, momentum, batch_size, num_epochs in product(
        learning_rates, momentums, batch_sizes, num_epochs_list
    ):
        dataset = PupilDataset("dataset", get_transform(train=False))
        experiment_name = (
            f"lr_{lr}_momentum_{momentum}_batch_{batch_size}_epochs_{num_epochs}"
        )
        writer = SummaryWriter("runs/" + experiment_name + "_" + current_time)

        # torch.manual_seed(277098)
        indices = torch.randperm(len(dataset)).tolist()
        if len(indices) == 0:
            raise ValueError("Empty dataset or indices issue.")
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size

        dataset, dataset_test = random_split(dataset, [train_size, test_size])

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"selected device {device}")
        print(f"is CUDA available?: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")

        num_classes = 2
        model = build_model(num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        # TODO
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )

        train_loss = train_one_epoch(
            model, optimizer, data_loader, device, num_epochs, print_freq=10
        )
        writer.add_scalar("Loss/train", train_loss, num_epochs)

        lr_scheduler.step()

        # Log the current learning rate
        current_lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar("Learning Rate", current_lr, num_epochs)

        evaluate(model, data_loader_test, device=device)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, optimizer, data_loader, device, epoch, print_freq=10
            )
            writer.add_scalar("Loss/train", train_loss, epoch)

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar("Learning Rate", current_lr, epoch)

            validation_loss = evaluate_loss(model, data_loader_test, device=device)
            writer.add_scalar("Loss/validation", validation_loss, epoch)

            evaluate(model, data_loader_test, device=device)  # validate loss

        torch.save(
            model,
            f"models/model_{current_time}.pt",
        )
        writer.close()
