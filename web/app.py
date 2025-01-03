import streamlit as st
import os
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'scripts'))

from yolo_prediction import predict_image

st.title("Metallic Probe Detection")
st.write("Upload an image, and the model will detect the metallic probe.")

# Model name input
model_name = 'YOLOv11_Small_Augmentation_Scratch' # st.text_input("Enter the model name:", value="my_yolo_model")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Detect"):
        with st.spinner('Processing...'):
            img_with_boxes, detected = predict_image(model_name, image)
            st.image(img_with_boxes, caption='Processed Image', use_column_width=True)

            if detected:
                st.success("Metallic probe detected!")
            else:
                st.warning("Warning: The model did not detect the metallic probe.")