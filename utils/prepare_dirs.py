import os

necessary_dirs = [
    "results/",
    "dataset/",
    "dataset/converted_images",
    "dataset/converted_masks",
    "snippets",
]

for dir in necessary_dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
