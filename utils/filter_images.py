import glob
import os
import shutil

import cv2
import numpy as np


def copy_images_and_masks(
    src_images_folder, src_masks_folder, dest_images_folder, dest_masks_folder, keyword
):
    os.makedirs(dest_images_folder, exist_ok=True)
    os.makedirs(dest_masks_folder, exist_ok=True)

    for root, dirs, files in os.walk(src_masks_folder):
        for file_name in files:
            if pupil_keyword in file_name:
                if keyword in file_name:
                    # Copy Mask
                    src_mask_path = os.path.join(root, file_name)
                    dest_mask_path = os.path.join(
                        dest_masks_folder, os.path.basename(src_mask_path)
                    )
                    shutil.copy(src_mask_path, dest_mask_path)

                    # Copy Image (assuming same name but with .jpg extension)
                    image_name = (
                        os.path.splitext(file_name)[0].replace("_pupil", "") + ".jpg"
                    )
                    src_image_path = os.path.join(
                        src_images_folder, os.path.basename(root), image_name
                    )
                    dest_image_path = os.path.join(dest_images_folder, image_name)
                    shutil.copy(src_image_path, dest_image_path)


if __name__ == "__main__":
    dataset_images_folder = (
        "C:\Studia\Semestr_9\Programowanie\GIT\PupilCheck-Pro\dataset\Images_converted"
    )
    dataset_masks_folder = (
        "C:\Studia\Semestr_9\Programowanie\GIT\PupilCheck-Pro\dataset\Masks"
    )
    dest_images_folder = (
        "C:\Studia\Semestr_9\Programowanie\GIT\PupilCheck-Pro\dataset\converted_images"
    )
    dest_masks_folder = (
        "C:\Studia\Semestr_9\Programowanie\GIT\PupilCheck-Pro\dataset\converted_masks"
    )
    pupil_keyword = "_pupil"
    keyword = "Rs"

    copy_images_and_masks(
        dataset_images_folder,
        dataset_masks_folder,
        dest_images_folder,
        dest_masks_folder,
        keyword,
    )

    keyword = "Ls"

    copy_images_and_masks(
        dataset_images_folder,
        dataset_masks_folder,
        dest_images_folder,
        dest_masks_folder,
        keyword,
    )
