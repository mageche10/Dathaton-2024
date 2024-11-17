import io

import requests
import streamlit as st
from PIL import Image
import requests

#st.image('\images.png')


def analyze_image(img_name):
    url = "http://127.0.0.1:5000/predictPhoto"
    params = {'img_name': img_name}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print('Error: ', response.status_code)


demo = st.toggle("Demo Activada")

st.title("AI Clothing Image Analyzer")
st.write("Upload or take clothing photos to analyze them.")

uploaded_files = st.file_uploader("Choose clothing images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

c = None
col1, col2 = st.columns([1, 2])
if col1.toggle("Activate camera"):
    c = col2.camera_input("", label_visibility="collapsed")
    ""

if c is not None:
    uploaded_files.append(c)

try:
    if uploaded_files:
        images = [file for file in uploaded_files]
        for photo in images:
            image = Image.open(photo)
            col1, col2 = st.columns([1, 2])

            with col1:
                if photo == c:
                    k = image.size
                    box = (k[0] / 2 - 120, k[1] / 2 - 168, k[0] / 2 + 120, k[1] / 2 + 168)
                    image = image.crop(box).resize((480, 672))
                st.image(image, caption=photo.name, use_container_width=True)

            with col2:
                if c is not None:
                    imggg = Image.open(io.BytesIO(c.getvalue()))
                    imggg.save('../../captured_photo.jpg')

                    with st.spinner('Analyzing'):
                        analysis = analyze_image('../../../captured_photo.jpg')
                elif demo:
                    with st.spinner('Analyzing'):
                        analysis = analyze_image('../../../../' + photo.name)
                else:
                    with st.spinner('Analyzing'):
                        analysis = analyze_image(photo.name)
                col3, col4 = st.columns([2, 1])

                with col3:
                    st.subheader("Analysis Result:")
                    st.write("Cane Height Type: " + analysis["cane_height_type"])
                    st.write("Closure Placement: " + analysis["closure_placement"])
                    st.write("Heel Shape Type: " + analysis["heel_shape_type"])
                    st.write("Knit Structure: " + analysis["knit_structure"])
                    st.write("Length Type: " + analysis["length_type"])

                with col4:
                    st.write("Neck Lapel Type: " + analysis["neck_lapel_type"])
                    st.write("Silhouette Type: " + analysis["silhouette_type"])
                    st.write("Sleeve Length Type: " + analysis["sleeve_length_type"])
                    st.write("Toecap Type: " + analysis["toecap_type"])
                    st.write("Waist Type: " + analysis["waist_type"])
                    st.write("Woven Structure: " + analysis["woven_structure"])
    else:
        "Please upload at least one image to start the analysis."
except:
    ""
