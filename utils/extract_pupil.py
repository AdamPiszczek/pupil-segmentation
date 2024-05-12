import glob
import os
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def process_image(image_path: str) -> np.ndarray:
    image_cv2: np.ndarray = cv2.imread(image_path)
    hsv: np.ndarray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    lower_blue: np.ndarray = np.array([90, 50, 50])
    upper_blue: np.ndarray = np.array([130, 255, 255])
    mask_blue: np.ndarray = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask_blue


def extract_pupil() -> None:
    images: List[str] = glob.glob("dataset/Masks/*/*.png")
    for image in tqdm(images, desc="Processing Images"):
        new_file_name: str = image.split(".png")[0] + "_pupil.png"
        if os.path.exists(new_file_name):
            print(f"File {new_file_name} already exists!")
            continue
        mask_blue: np.ndarray = process_image(image)
        cv2.imwrite(new_file_name, mask_blue)


if __name__ == "__main__":
    extract_pupil()
