import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

data_filepath = "./data/"

image_width = 160
image_height = 224

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


one_hot_df = pd.get_dummies(product_data["des_product_aggregated_family"], prefix="agg_fam_").astype("int32")
product_data[one_hot_df.columns] = one_hot_df
product_data = product_data.drop("des_product_aggregated_family", axis = 1)

one_hot_list = list(one_hot_df.to_dict("index").values())


images = []
c = 0
for filename in product_data["des_filename"]:
    oh = np.transpose(np.array(list((one_hot_list[c]).values())))

    #grayscale, normalize and reduce data for easier treatment
    img = Image.open("./data/images/images_fr/" + filename).convert('L')
    np_img = np.transpose(np.array(img))
    np_img = zoom(np_img, (1/red_factor, 1/red_factor))
    np_img = np.divide(np_img, 255)

    np_all = np.concatenate((np_img, np.tile(oh, (int(image_width / red_factor), 1))), axis=1)
    images.append(tf.convert_to_tensor(np_all))
    c += 1
    if(c % 100 == 0):
        print(c)




#list of attributes for which the program will train models
attribute_list = ["silhouette_type"]

#create and train model for each attribute
for attr in attribute_list:
    dataset_dir = "./data/images/"

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


    #set up datasets from images
    sp_index = int(0.8 * len(images))

    train_dataset = tf.stack(images[:sp_index])
    validation_dataset = tf.stack(images[sp_index:])

    print(train_dataset.shape)
    print(validation_dataset.shape)





    image_array = np.array(images)  # Shape: (n, height, width)
    label_array = np.array(labels[:len(images)])  # Shape: (n,)

    # Expand dimensions of images to add the channel dimension (grayscale)
    image_array = image_array[..., np.newaxis]  # Shape: (n, height, width, 1)

    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((image_array, label_array))
    dataset = dataset.shuffle(buffer_size=1000).batch(32) 

    #create model
    model = keras.Sequential([
        layers.InputLayer(input_shape=(image_array.shape[1], image_array.shape[2], 1)), 
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_set), activation='softmax') 
    ])

    #compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #fit model
    model.fit(
        dataset,
        epochs=8  #adjust as needed
    )
    #save model for later use
    model.save("./models/" + attr + ".keras")


print("ok")