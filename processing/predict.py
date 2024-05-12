import glob
import os
import random
import traceback  # Uncomment for debugging purposes

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CLASS_NAMES = ["__background__", "pupil"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_coloured_mask(mask):
    if mask.shape != (1700, 3000):
        return mask
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    indices = np.where(mask == 1)
    if indices[0].size > 0:  # Check if there are any True values in the mask
        chosen_colour = colours[random.randrange(0, 10)]
        r[indices] = chosen_colour[0]
        g[indices] = chosen_colour[1]
        b[indices] = chosen_colour[2]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    pred_array = [pred_score.index(x) for x in pred_score if x > confidence]
    if len(pred_array) == 0:
        raise ValueError("No object have been found!")
    pred_t = pred_array[-1]
    masks = (
        (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    )  # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]
    masks = masks[: pred_t + 1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return masks, pred_boxes, pred_class


def segment_instance(
    img_path, output_dir="./results/", confidence=0.5, rect_th=2, text_size=2, text_th=2
):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    output_file = (
        img_path.split(".jpg")[0].split("converted_images")[1] + "_converted.png"
    )
    output_file = output_dir + output_file
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    img = cv2.imread(img_path)
    for i in range(len(masks)):
        if masks[i].shape != (1700, 3000):
            raise ValueError("No object have been found!")
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))  # Top-left corner
        pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        text_position = (pt1[0], pt1[1] - 10)
        cv2.putText(
            img,
            pred_cls[i],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 255, 0),
            thickness=text_th,
        )
    cv2.imwrite(output_file, img)


if __name__ == "__main__":
    output_dir = "./results/predict/"
    jpg_files = glob.glob("./dataset/converted_images/*.jpg")
    converted_images = glob.glob(output_dir + "*_converted.png")

    if len(converted_images) > 0:
        print(
            f"There are already converted images in {output_dir} dir, do you want to proceed and overwrite this files? ('Y/y' or 'N/n')"
        )
        user_input = input().lower()
        if user_input == "y":
            for image in converted_images:
                os.remove(image)
            pass
        elif user_input == "n":
            exit(0)
        else:
            print("Invalid input. Please enter 'y', 'n'")
            exit(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}")
    model_path = "models/model_20240120-161444.pt"
    model = torch.load(model_path)
    print(f"Model {model_path} have been succsesfully loaded!")
    model.eval()
    model.to(device)

    confidence = 0.1  # 0.0 - 1.0
    counter = 0
    for file in tqdm(jpg_files, desc="Processing images"):
        try:
            segment_instance(file, output_dir=output_dir, confidence=confidence)
        except Exception as e:
            # print(e) # Uncomment for debugging purposes
            # traceback.print_exc() # Uncomment for debugging purposes
            counter += 1

    print(
        f"Model did not find any pupil in image in {counter} out of {len(jpg_files)} images, with confidence {confidence*100}%"
    )
