from ultralytics import YOLO
import os
import torch
import numpy as np
import tempfile
from PIL import Image


def predict_image(model_name, image):
    model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'yalo', model_name, 'weights', 'best.pt')
    model = YOLO(model_path)
    
    device = 'cpu' # 0 if torch.cuda.is_available else 'cpu'
        
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
    
    for r in result:
        if r.boxes:
            detected = True
            # Draw bounding boxes on the image
            img_with_boxes = r.plot()
            # Remove temporary image file
            os.remove(img_path)
            return img_with_boxes, detected
        else:
            # If no detections, return original image
            img = np.array(image)
            # Remove temporary image file
            os.remove(img_path)
            return img, detected