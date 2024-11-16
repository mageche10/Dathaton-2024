import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

data_filepath = "./data/"

image_width = 160
image_height = 224

train_dataset = image_dataset_from_directory(
    data_filepath,
    labels = None,
    seed = 1234567
)

# process:
# identify type of clothing
# identify properties

product_dataset = pd.read_csv(data_filepath + "product_data.csv")

primary_classifier = tf.keras.models.Sequential()
model.add(Dense(num, input_shape=(len(features),)))
model.add(Dense(num, activation = 'func'))