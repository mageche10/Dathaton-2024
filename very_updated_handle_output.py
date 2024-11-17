import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import os
import re
import sys
from scipy.ndimage import zoom

attributes = ["silhouette_type"]
models = {}

for attr in attributes:
    models[attr] = keras.models.load_model("./models/very_updated_" + attr + ".keras")

prediction_dir = "./data/images/images_test/"
images = os.listdir(prediction_dir)
test_df = pd.read_csv("./data/test_data.csv")
red_factor = 4
predicting_columns = ["des_sex", "des_line", "des_fabric", "des_product_family", "des_product_type"]

test_df[predicting_columns] = test_df[predicting_columns].astype("category")
test_df[predicting_columns] = test_df[predicting_columns].apply(lambda x: x.cat.codes)

for attr in attributes:
    pred = []
    count = 0
    for img in images:
        current_df = test_df[test_df["des_filename"] == img]
        current_df = current_df.head(1)[predicting_columns]
        current_df = current_df

        #grayscale, normalize and reduce data for easier treatment
        img_layer = Image.open("./data/images/images_test/" + img).convert('L')
        np_img = np.transpose(np.array(img_layer))
        np_img = zoom(np_img, (1/red_factor, 1/red_factor))
        np_img = np.divide(np_img, 255)
        img_array = np_img[np.newaxis, ..., np.newaxis]
        pred.append(np.argmax((models[attr]).predict([img_array, current_df], verbose=0) , axis=1))
        count += 1
        if count % 100 == 0:
            print(count, " / ", len(images))
    pred.to_csv("./predictions/very_updated_" + attr + ".csv", index = False)
