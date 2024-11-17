from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import pandas as pd

# Crear la aplicación Flask
app = Flask(__name__)


# Ruta raíz
@app.route('/')
def home():
    return "API funcionando localmente"


@app.route('/predictPhoto', methods=['GET'])
def predictPhoto():
    img_name = request.args.get('img_name')
    img_path = './data/images/images_test/' + img_name

    attributes_set = pd.read_csv('./data/attribute_data.csv')


    labels = []
    atr = ['silhouette_type', 'neck_lapel_type', 'woven_structure', 'knit_structure', 'heel_shape_type', 'length_type',
           'sleeve_length_type', 'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type']
    for a in atr:
        all_values = attributes_set[attributes_set["attribute_name"] == a][['cod_value', 'des_value']]
        non_duplicated = all_values.drop_duplicates(subset=['cod_value']).sort_values(by=['cod_value'])
        values_list = non_duplicated['des_value'].tolist()

        model = keras.models.load_model("./models/" + a + ".keras")
        img = image.load_img(img_path, target_size=(160, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model input shape
        normalization_layer = layers.Rescaling(1. / 255)
        img_array = normalization_layer(img_array)  # Normalize the image

        # Predict the label
        pred = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(pred, axis=1)[0]  # Get the class index with the highest probability
        labels.append(values_list[predicted_label-1] if predicted_label != 0 else 'INVALID')

    resposta = {
        "silhouette_type": labels[0],
        "neck_lapel_type": labels[1],
        "woven_structure": labels[2],
        "knit_structure": labels[3],
        "heel_shape_type": labels[4],
        "length_type": labels[5],
        "sleeve_length_type": labels[6],
        "toecap_type": labels[7],
        "waist_type": labels[8],
        "closure_placement": labels[9],
        "cane_height_type": labels[10],
    }
    return jsonify(resposta), 200

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
