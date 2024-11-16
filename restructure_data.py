import pandas as pd
import shutil
import os

from PIL import Image

dataset_dir = "./data/images/images_fr"
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img.verify()  # Verify the image is not corrupt
            except (IOError, SyntaxError) as e:
                print(f"Corrupt image found: {img_path}")