# Probe Detection Project

This project involves training, evaluating, and deploying multiple YOLOv11 object detection models. The models are trained on a custom dataset with different configurations, evaluated on test data, and deployed through a Streamlit application for real-time predictions.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)

---

## Project Structure
``` 
├── data
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── val
│   │   ├── images
│   │   └── labels
│   └── test
│       ├── images
│       └── labels
├── models
│   ├── YOLOv11_Small_No_Augmentation_Scratch
│   ├── YOLOv11_Small_Augmentation_Scratch
│   ├── YOLOv11_Small_No_Augmentation_PreTrained
│   ├── YOLOv11_Small_Augmentation_PreTrained
│   ├── YOLOv11_Large_No_Augmentation_Scratch
│   ├── YOLOv11_Large_Augmentation_Scratch
│   ├── YOLOv11_Large_No_Augmentation_PreTrained
│   └── YOLOv11_Large_Augmentation_PreTrained
├── scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── web
│   └──app.py
├── requirements.txt
└── README.md
```

### Folders

- `data/`: Contains the dataset divided into training (`train/`), validation (`val/`), and test (`test/`) sets. Each set contains `images/` and `labels/` directories
- `models/`: Contains directories for each trained model configuration along with their weights
- `scripts/`: Contains Python scripts
- `app.py`: The Streamlit application for running the model in real-time
- `requirements.txt`: Lists all the dependencies required to run the project

---

## Data Preparation

The dataset is organized into three subsets:

1. **Training Set (`data/train/`):** Used for training the models.
2. **Validation Set (`data/val/`):** Used for validating the model during training.
3. **Test Set (`data/test/`):** Used for evaluating the final model performance.

Each subset contains:

- `images/`: The images used for training/testing.
- `labels/`: Corresponding annotation files in YOLO format.

---

## Model Training

Eight different YOLOv11 models are trained with varying configurations:

1. **Size Variants:**
   - Small
   - Large

2. **Data Augmentation:**
   - With Augmentation
   - Without Augmentation

3. **Initialization:**
   - Trained from Scratch
   - Pre-Trained Weights


## Model Evaluation

Models are evaluated on the test dataset to compare performance metrics and inference speed.

### Performance Metrics

The key metrics used for evaluation include:

- **Precision:** The ratio of true positives to the sum of true and false positives.
- **Recall:** The ratio of true positives to the sum of true positives and false negatives.
- **mAP@50:** Mean Average Precision at an IoU threshold of 0.5.
- **mAP@50-95:** Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.

---

## Streamlit Application

The `app.py` file contains the Streamlit application, which allows users to:

- Upload an image.
- Select a model for prediction.
- View the detection results with bounding boxes.
- See whether objects were detected in the image.

### Running the App

Instructions on how to run the app : 

```
cd web/
streamlit run app.py
```
