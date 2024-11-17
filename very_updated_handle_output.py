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

def get_invalid_pairs():
    pr_df = pd.read_csv("./data/product_data.csv")
    at_df = pd.read_csv("./data/attribute_data.csv")

    attrs = at_df["attribute_name"].unique()
    types = pr_df["des_product_type"].unique()

    print(attrs)
    print(types)

    tdict = {}
    adict={}

    count = 0
    for t in pr_df["cod_modelo_color"]:
        tdict[t] = pr_df.loc[count, "des_product_type"]
        count += 1

    count = 0

    for a in attrs:
        adict[a] = set()
    for a in at_df["attribute_name"]:
        adict[a].add(tdict[at_df.loc[count, "cod_modelo_color"]])
        count += 1

    pairs = []

    for a in attrs:
        for t in types:
            if not (t in adict[a]):
                pairs.append([t, a])

    return pairs

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
    attributes_set = pd.read_csv('./data/attribute_data.csv')
    attributes_names = attributes_set["attribute_name"].unique()

    all_values = attributes_set[attributes_set["attribute_name"] == attr][['cod_value', 'des_value']]
    non_duplicated = all_values.drop_duplicates(subset=['cod_value']).sort_values(by=['cod_value'])

    values_list = non_duplicated['des_value'].tolist()
    predictions = []
    image_names = []

    count = 0
    for img in images:
        if count > 100:
            break
        current_df = test_df[test_df["des_filename"] == img]
        if [current_df.head(1)["des_product_type"], attr] in pairs:
            current_prediction = 0
            predictions.append(current_prediction) 
            image_names.append(img)
            continue

        current_df = current_df.head(1)[predicting_columns]
        current_df = current_df

        #grayscale, normalize and reduce data for easier treatment
        img_layer = Image.open("./data/images/images_test/" + img).convert('L')
        np_img = np.transpose(np.array(img_layer))
        np_img = zoom(np_img, (1/red_factor, 1/red_factor))
        np_img = np.divide(np_img, 255)
        img_array = np_img[np.newaxis, ..., np.newaxis]

        current_prediction = np.argmax((models[attr]).predict([img_array, current_df], verbose=0) , axis=1)[0]


        predictions.append(current_prediction) 
        image_names.append(img)

        count += 1
        if count % 100 == 0:
            print(count, " / ", len(images))
    
    submission_dataframe = pd.DataFrame(columns=["test_id", "des_value"])
    j = 0
    for img_name, pred_label in zip(image_names, predictions):
        #print(f"Image: {img_name} - Predicted Label: {pred_label}")
        pattern = r"^\d+_\d+"

        submission_dataframe.loc[j] = [re.match(pattern, img_name).group() + '_' + attr,
                                  values_list[pred_label - 1] if pred_label != 0 else 'INVALID']
        j = j + 1
    submission_dataframe.to_csv("./predictions/very_updated_" + attr + ".csv", index = False)
