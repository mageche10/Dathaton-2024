import pandas as pd
import shutil
import os

test_images = pd.read_csv("./data/product_data.csv")["des_filename"]
count = 0
for filename in test_images:
    shutil.copyfile("./data/images/images/" + filename, "./data/images/images_fr/" + filename)
    count += 1
    if(count % 200 == 0):
        print(count)