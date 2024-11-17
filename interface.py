import streamlit as st
from PIL import Image
import keras
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

st.image('\images.png')

def analyze_image(img_path):
    labels=[]
    atr=['silhouette_type', 'neck_lapel_type', 'woven_structure', 'knit_structure', 'heel_shape_type', 'length_type', 'sleeve_length_type', 'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type']
    for a in atr:
      model=keras.models.load("./models/"+a+".keras")
      img = image.load_img(img_path, target_size=(160,224))
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model input shape
      normalization_layer=layers.Reescaling(1./255)
      img_array = normalization_layer(img_array)  # Normalize the image

      # Predict the label
      pred = model.predict(img_array, verbose=0)
      predicted_label = np.argmax(pred, axis=1)[0]  # Get the class index with the highest probability
      labels.append(predicted_label)
    print(labels)
    

st.title("AI Clothing Image Analyzer")
st.write("Upload or take clothing photos to analyze them.")

uploaded_files = st.file_uploader("Choose clothing images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

c=None
col1, col2 = st.columns([1,2])
if(col1.toggle("Activate camera")):
  c=col2.camera_input("",label_visibility="collapsed")
  ""

if(c!=None):
  uploaded_files.append(c)


try:
  if uploaded_files:
    images = [file for file in uploaded_files]
    i=0
    for photo in images:
      image = Image.open(photo)
      link="./data/images/images/"+photo.name
      if(photo==c):
        k=image.size
        box=(k[0]/2-120,k[1]/2-168,k[0]/2+120,k[1]/2+168)
        image=image.crop(box).resize((160,224))
      col1, col2= st.columns([1,2])
            
      with col1:
        st.image(image, caption=photo.name, use_container_width=True)
            
      with col2 :
        analysis_result = analyze_image(link)
        print(analysis_result)
        st.subheader("Analysis Result:")
        st.write("")
  else:
    "Please upload at least one image to start the analysis."
except:
  ""