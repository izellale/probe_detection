import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'scripts'))

from yolo_prediction import predict_image

st.title("Metallic Probe Detection")
st.write("Upload an image, and the model will detect the metallic probe.")

@st.cache_resource
def load_model():
    model_name = 'YOLOv11_Small_Augmentation_PreTrained'
    model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'yalo', model_name, 'weights', 'best.pt')
    return YOLO(model_path)

yolo_model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Predict button
    if st.button("Detect"):
        with st.spinner('Processing...'):
            img_with_boxes, detected, inference_speed = predict_image(yolo_model, image)
            
            if detected:
                st.success(f"Metallic probe detected in {inference_speed:.2f} ms !", icon="✅")
                st.image(img_with_boxes, caption='Image with Detected Probe', use_container_width=True)
            else:
                st.error("The model did not detect the metallic probe.", icon="🚨")
                st.image(img_with_boxes, caption='Original Image', use_container_width=True)