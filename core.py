import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

data_filepath = "./data/"

image_width = 160
image_height = 224


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

#initialize the dictionary holding the encoding for every 
for attr in attribute_list:
    attribute_image_dict[attr] = {}
    for img in individual_image_list:
        attribute_image_dict[attr][img] = 0


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

#list of attributes for which the program will train models
attribute_list = ["length_type"]

#create and train model for each attribute
for attr in attribute_list:
    dataset_dir = "./data/images/images_fr"

    #encode labels
    labels = []
    for img in individual_image_list:
        labels.append(int(attribute_image_dict[attr][img]))


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


    #set up datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,  
        label_mode='int',   
        batch_size=32,      
        image_size=(image_width, image_height), 
        validation_split=0.2, 
        subset="training",      
        seed=123456
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,
        label_mode='int',
        batch_size=32,
        image_size=(image_width, image_height), 
        validation_split=0.2,
        subset="validation",
        seed=123456
    )

    #normalize data
    normalization_layer = layers.Rescaling(1./255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


    #create model
    model = keras.Sequential([
        layers.InputLayer(input_shape=(image_width, image_height, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_set), activation='softmax') #as many outputs as categories
    ])

    #compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #fit model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=3  # Adjust the number of epochs as needed
    )

    #save model for later use
    model.save("./models/" + attr + ".keras")


print("ok")