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


def make_submission(name):
    attributes_set = pd.read_csv('./data/attribute_data.csv')
    attributes_names = attributes_set["attribute_name"].unique()

    all_values = attributes_set[attributes_set["attribute_name"] == name][['cod_value', 'des_value']]
    non_duplicated = all_values.drop_duplicates(subset=['cod_value']).sort_values(by=['cod_value'])

    values_list = non_duplicated['des_value'].tolist()

    model = keras.models.load_model('./models/' + name + '.keras')
    # Directory containing images for prediction
    prediction_dir = "./data/images/images_test"

    # Preprocess and predict images
    image_size = (160, 224)  # Must match the size used during training
    normalization_layer = layers.Rescaling(1. / 255)

    predictions = []
    image_names = []

    n = 0
    images = os.listdir(prediction_dir)
    for img_filename in images:
        img_path = os.path.join(prediction_dir, img_filename)
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
            # Load the image and resize it
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model input shape
            img_array = normalization_layer(img_array)  # Normalize the image

            # Predict the label
            pred = model.predict(img_array, verbose=0)
            predicted_label = np.argmax(pred, axis=1)[0]  # Get the class index with the highest probability

            # Append to results
            predictions.append(predicted_label)
            image_names.append(img_filename)

            print(str(n) + '/' + str(len(images)))
            n = n + 1

    submission_dataframe = pd.DataFrame(columns=["test_id", "des_value"])
    # Print or save the predictions
    j = 0
    for img_name, pred_label in zip(image_names, predictions):
        print(f"Image: {img_name} - Predicted Label: {pred_label}")
        pattern = r"^\d+_\d+"

        submission_dataframe.loc[j] = [re.match(pattern, img_name).group() + '_' + name,
                                  values_list[pred_label - 1] if pred_label != 0 else 'INVALID']
        j = j + 1

    return submission_dataframe

ls = ["cane_height_type", "closure_placement", "waist_type", "toecap_type", "sleeve_length_type"]

for attr in ls:
    pred = make_submission(attr)
    pred.to_csv("./predictions/" + attr + ".csv", index = False)
