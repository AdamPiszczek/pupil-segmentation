import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def calculate_center_white_percentage(image_cv2: np.ndarray) -> bool:
    gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    center_start_x = int(width * 0.4)
    center_end_x = int(width * 0.6)
    center_start_y = int(height * 0)
    center_end_y = int(height * 1)
    center_region = gray_image[center_start_y:center_end_y, center_start_x:center_end_x]

    white_pixels = cv2.countNonZero(center_region)
    return white_pixels > 0


images = glob.glob("dataset/Masks/*/*_pupil.png")
counter = 0
counter2 = 0

for image_path in tqdm(images, desc="Processing Images"):
    image_cv2 = cv2.imread(image_path)

    # --- check how area looks like ---
    # height, width, _ = image_cv2.shape
    # # center_start_x = int(width * 0.25)
    # # center_end_x = int(width * 0.75)
    # # center_start_y = int(height * 0)
    # # center_end_y = int(height * 1)
    # cv2.rectangle(image_cv2, (center_start_x, center_start_y), (center_end_x, center_end_y), (0, 0, 255), 2)
    # cv2.imwrite('results/test.png', image_cv2)
    # break

    if calculate_center_white_percentage(image_cv2):
        counter += 1
        corresponding_image = (
            image_path.split("_pupil.png")[0].replace("Masks", "Images") + ".jpg"
        )
        if os.path.exists(corresponding_image):
            destination_folder = "dataset/converted_images/"
            shutil.copy(corresponding_image, destination_folder)
            destination_folder = "dataset/converted_masks/"
            shutil.copy(image_path, destination_folder)
    else:
        counter2 += 1

print(
    f"Total of {counter} images were copied to dataset/converted* folder, omitted {counter2} files"
)
