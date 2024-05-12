import glob
import os
import shutil

import cv2
import numpy as np


def check_histogram_for_brightness(src_image_path):
    # Image load
    img = cv2.imread(src_image_path)

    # Convert img into grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # Mean brightness
    brightness = np.mean(hist)

    return brightness


# Path to image folder
path = "C:\Studia\Semestr_9\Programowanie\GIT\PupilCheck-Pro\dataset\converted_images"


def Mean_brightness(path):
    mean_brightness = 0
    brightness = []
    images = glob.glob(path + "/*.jpg")
    for image in images:
        # print(image)
        bright = check_histogram_for_brightness(os.path.join(path, image))
        brightness.append(bright)
        # print("Brightness of image " + image + " is " + str(bright))
        mean_brightness += int(bright)
    mean_brightness = mean_brightness / len(brightness)
    return mean_brightness


print(Mean_brightness(path))
