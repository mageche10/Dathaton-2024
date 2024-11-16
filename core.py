import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

data_filepath = "./data/"

pd.options.display.max_columns = None

image_width = 160
image_height = 224


attribute_data = pd.read_csv(data_filepath + "attribute_data.csv")
product_data = pd.read_csv(data_filepath + "product_data.csv")

print(product_data.shape)

#corrupted images that must be removed
corrupted_files_condition = (product_data["des_filename"] == "83_1148656_17026323-07_B.jpg") | (product_data["des_filename"] == "86_1208032_47001267-15_.jpg")
product_data = product_data.drop(product_data[corrupted_files_condition].index)

print(product_data.shape)

attribute_list = list(attribute_data["attribute_name"].unique())

product_data = product_data.sort_values(["cod_modelo_color", "des_filename"])
attribute_data = attribute_data.sort_values(["cod_modelo_color", "attribute_name"])

attribute_labels = {}

biglist = list(product_data["cod_modelo_color"])

megadict = {}

for attr in attribute_list:
    megadict[attr] = {}
    for img in biglist:
        megadict[attr][img] = 0

print("ok???")

for attr in attribute_list:
    column = attribute_data[attribute_data["attribute_name"] == attr].copy()  
    
    column[attr] = column["cod_value"].astype("object")
    ncolumn = column[["cod_modelo_color", attr]]
    
    this_dict = ncolumn.to_dict()
    this_list1 = list(this_dict["cod_modelo_color"].values())
    this_list2 = list(this_dict[attr].values())

    i = -1
    for img in this_list1:
        i += 1
        megadict[attr][this_list1[i]] = this_list2[i]



attribute_list = ["length_type"]


for attr in attribute_list:
    dataset_dir = "./data/images/images_fr"
    labels = []
    for img in biglist:
        labels.append(int(megadict[attr][img]))


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


    normalization_layer = layers.Rescaling(1./255)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,  # Automatically infers labels based on subdirectory names
        label_mode='int',   # You can use 'categorical' if you need one-hot encoding
        batch_size=32,      # Number of images per batch
        image_size=(160, 224),  # Resize images to this size
        validation_split=0.2,   # Split 20% for validation
        subset="training",      # This subset is for training
        seed=123                # Seed for reproducibility
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,
        label_mode='int',
        batch_size=32,
        image_size=(160, 224),
        validation_split=0.2,
        subset="validation",
        seed=123
    )


    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


    # Step 3: Create the model
    model = keras.Sequential([
        layers.InputLayer(input_shape=(image_width, image_height, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(label_set), activation='softmax')  # Adjust the number of classes as needed
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if using one-hot encoding
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=3  # Adjust the number of epochs as needed
    )

    model.save("./models/" + attr + ".keras")


print("ok")