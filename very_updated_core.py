import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import sys

data_filepath = "./data/"

image_width = 160
image_height = 224

maxNum = 40000

red_factor = 4

attribute_data = pd.read_csv(data_filepath + "attribute_data.csv")
product_data = pd.read_csv(data_filepath + "product_data.csv")

#corrupted images that must be removed from the dataframe
corrupted_files_condition = (product_data["des_filename"] == "83_1148656_17026323-07_B.jpg") | (product_data["des_filename"] == "86_1208032_47001267-15_.jpg")
product_data = product_data.drop(product_data[corrupted_files_condition].index)

#list of attributes to guess
attribute_list = list(attribute_data["attribute_name"].unique())

#sort dataframes for comfort
product_data = product_data.sort_values(["cod_modelo_color", "des_filename"])
attribute_data = attribute_data.sort_values(["cod_modelo_color", "attribute_name"])

individual_image_list = list(product_data["cod_modelo_color"])

attribute_image_dict = {}

product_data = product_data.drop(["cod_color", "des_age", "des_product_category", "des_product_aggregated_family", "des_color"], axis = 1)

#initialize the dictionary holding the encoding for every label
for attr in attribute_list:
    attribute_image_dict[attr] = {}
    for img_layer in individual_image_list:
        attribute_image_dict[attr][img_layer] = 0


#create labels
for attr in attribute_list:
    column = attribute_data[attribute_data["attribute_name"] == attr].copy()  
    
    column[attr] = column["cod_value"].astype("object")
    column = column[["cod_modelo_color", attr]]
    
    this_dict = column.to_dict()
    this_list1 = list(this_dict["cod_modelo_color"].values())
    this_list2 = list(this_dict[attr].values())

    for i in range(0, len(this_list1)):
        attribute_image_dict[attr][this_list1[i]] = this_list2[i]

#give categorical features their encoding

for column in product_data.columns:
    if str(column) != "cod_modelo_color" and str(column) != "des_filename":
        product_data[column] = product_data[column].astype('category')

categorical_columns = product_data.select_dtypes(['category']).columns
product_data[categorical_columns] = product_data[categorical_columns].apply(lambda x: x.cat.codes)
print(product_data.head())

images = []
c = 0
for filename in product_data["des_filename"]:
    #grayscale, normalize and reduce data for easier treatment
    img_layer = Image.open("./data/images/images_fr/" + filename).convert('L')
    np_img = np.transpose(np.array(img_layer))
    np_img = zoom(np_img, (1/red_factor, 1/red_factor))
    np_img = np.divide(np_img, 255)
    images.append(np_img)
    c += 1
    if(c >= maxNum):
        break
    if(c % 100 == 0):
        print(c)




#list of attributes for which the program will train models
attribute_list = ["silhouette_type"]

#create and train model for each attribute
for attr in attribute_list:
    dataset_dir = "./data/images/"

    #encode labels
    labels = []
    for img_layer in individual_image_list:
        labels.append(int(attribute_image_dict[attr][img_layer]))


    label_set = set()
    for label in labels:
        label_set.add(label)
    
    count = 0
    label_dict = {}
    for label in label_set:
        label_dict[label] = count
        count += 1

    count = 0
    for label in labels:
        labels[count] = label_dict[labels[count]]
        count += 1



    image_array = np.array(images)  
    image_array = image_array[..., np.newaxis] #shape: (n, height, width, 1 (channel))
    label_array = np.array(labels[:len(images)]) 


    # Expand dimensions of images to add the channel dimension (grayscale)
    image_array = image_array[..., np.newaxis]  # Shape: (n, height, width, 1)

    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((image_array, label_array))

    #create model
    img_input = layers.Input(shape=(int(image_width / red_factor), int(image_height / red_factor), 1))
    data_input = layers.Input(shape = (len(product_data.columns) - 2,))
    img_layer = layers.Conv2D(64, (3, 3), activation='relu')(img_input)
    img_layer = layers.MaxPooling2D(pool_size=(2, 2))(img_layer)
    img_layer = layers.Conv2D(64, (3, 3), activation='relu')(img_layer)
    img_layer = layers.MaxPooling2D(pool_size=(2, 2))(img_layer)
    img_layer = layers.Conv2D(32, (3, 3), activation='relu')(img_layer)
    img_layer = layers.Flatten()(img_layer)
    merged_layers = layers.Concatenate()([img_layer, data_input])
    merged_layers = layers.Dense(32, activation='relu')(merged_layers)
    output = layers.Dense(len(label_set), activation='softmax') (merged_layers)

    model = keras.Model(inputs=[img_input, data_input], outputs = output)


    #compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    input_data = product_data.drop(["cod_modelo_color", "des_filename"], axis=1)[:maxNum]
    #fit model
    model.fit([image_array, input_data], label_array, batch_size=16, epochs=8)

    #save model for later use
    model.save("./models/very_updated_" + attr + ".keras")
