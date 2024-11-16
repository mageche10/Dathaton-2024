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


def load_image(image_path):
    image = Image.open(image_path)  
    #print("loading some")  
    image = image.convert('RGB')

    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    return image_tensor

def load_images_from_attribute(attr, attr_data, prod_data):
    tensors = []
    counter = -1
    for image_id in attr_data["cod_modelo_color"]:
        counter += 1
        if attr == attr_data.loc[counter, "attribute_name"]:
            for filename in prod_data[prod_data["cod_modelo_color"].isin([image_id])]["des_filename"]:
                try:
                    image_path = data_filepath + "images/images_sample/" + filename
                    tensor = load_image(image_path)
                    tensors.append(tensor)
                except Exception as e:
                    #print(f"Failed to load image {image_path}: {e}")
                    jo = "fdcsf"
    return tensors


attribute_data = pd.read_csv(data_filepath + "attribute_data.csv")
product_data = pd.read_csv(data_filepath + "product_data.csv")
#images_dataset = load_images_from_attribute("silhouette_type", attribute_data, product_data)

attribute_list = list(attribute_data["attribute_name"].unique())

product_data = product_data.sort_values(["cod_modelo_color", "des_filename"])
attribute_data = attribute_data.sort_values(["cod_modelo_color", "attribute_name"])

attribute_labels = {}

biglist = list(product_data["cod_modelo_color"])

megadict = {}

full_df = product_data
for attr in attribute_list:
    full_df[attr] = "none"
    megadict[attr] = {}
    for img in biglist:
        megadict[attr][img] = "none"

print("ok???")

for attr in attribute_list:
    # Filter the relevant rows
    column = attribute_data[attribute_data["attribute_name"] == attr].copy()  # Make a copy to avoid modifying a slice
    
    # Cast the 'cod_value' column to 'object' type
    column[attr] = column["cod_value"].astype("object")
    
    # Keep only the relevant columns for merging
    ncolumn = column[["cod_modelo_color", attr]]
    
    #print(ncolumn.head())
    this_dict = ncolumn.to_dict()
    '''if attr == "silhouette_type":
        print("es" , len(list(this_dict["cod_modelo_color"])))
        for i in range(0, 10):
            print(list(this_dict["cod_modelo_color"])[i])'''
    this_list1 = list(this_dict["cod_modelo_color"].values())
    this_list2 = list(this_dict[attr].values())

    i = -1
    for img in this_list1:
        i += 1
        megadict[attr][this_list1[i]] = this_list2[i]



attribute_list = ["silhouette_type"]


'''
counter = -1
for image_id in attribute_data["cod_modelo_color"]:
    
    counter += 1
    attr = attribute_data.loc[counter, "attribute_name"]
    condition = (full_df["cod_modelo_color"] == image_id)
    full_df.loc[condition, attr] = attribute_data.loc[counter, "cod_value"]
    if counter % 100 == 0:
        print(counter)
'''


'''for attr in attribute_list:
    current_labels = []
    for image_id in product_data["cod_modelo_color"]:
        condition = (attribute_data["cod_modelo_color"] == image_id) & (str(attribute_data["attribute_name"]) == str(attr))
        filtered_data = attribute_data[condition]
        if not filtered_data.empty: 
            current_labels.append(str(filtered_data))
        else:
            current_labels.append("invalid")


for i in range(0, 10):
    print(attribute_labels[0][i])
'''


for attr in attribute_list:
    dataset_dir = "./data/images/images_fr"
    labels = []
    for img in biglist:
        labels.append(str(megadict[attr][img]))





    normalization_layer = layers.Rescaling(1./255)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,  # Automatically infers labels based on subdirectory names
        label_mode='categorical',   # You can use 'categorical' if you need one-hot encoding
        batch_size=32,      # Number of images per batch
        image_size=(160, 224),  # Resize images to this size
        validation_split=0.2,   # Split 20% for validation
        subset="training",      # This subset is for training
        seed=123                # Seed for reproducibility
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels=labels,
        label_mode='categorical',
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
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')  # Adjust the number of classes as needed
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if using one-hot encoding
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=5  # Adjust the number of epochs as needed
    )


print("ok")