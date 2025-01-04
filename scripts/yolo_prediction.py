from ultralytics import YOLO
import os
import torch
import numpy as np
import tempfile
from PIL import Image


def predict_image(model, image):
    device = 0 if torch.cuda.is_available else 'cpu'
        
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img_path = tmp_file.name
        image.save(img_path)

    # Run prediction
    result = model.predict(
        source=img_path,
        conf=0.7,
        device=device
    )
    
    # Initialize detected flag
    detected = False
    inference_speed = None
    
    for r in result:
        if r.boxes:
            detected = True
            
            img_with_boxes = r.plot(labels=False)
            inference_speed = r.speed['inference']
            
            # Remove temporary image file
            os.remove(img_path)
            return img_with_boxes, detected, inference_speed
        else:
            # If no detections, return original image
            img = np.array(image)
            # Remove temporary image file
            os.remove(img_path)
            return img, detected, inference_speed