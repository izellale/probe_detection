import os
import json
from shutil import copyfile, move
from sklearn.model_selection import train_test_split


def convert_labels(input_json, images_dir, output_dir_images, output_dir_labels):
    with open(input_json) as f:
        data = json.load(f)

    annotations = data['annotations']
    images_info = {img['id']: img for img in data['images']}

    for annotation in annotations:
        bbox = annotation['bbox']
        image_id = annotation['image_id']
        image_info = images_info[image_id]

        # Extract bounding box coordinates
        x_min, y_min, width, height = bbox
        x_center = (x_min + (width / 2)) / image_info['width']
        y_center = (y_min + (height / 2)) / image_info['height']
        norm_width = width / image_info['width']
        norm_height = height / image_info['height']

        # Rename and copy the image to the new directory
        new_image_name = f"{image_id}.jpg"
        copyfile(os.path.join(images_dir, image_info['file_name']), os.path.join(output_dir_images, new_image_name))

        # Create label file with the same ID name
        label_filename = f"{image_id}.txt"
        with open(os.path.join(output_dir_labels, label_filename), 'w') as label_file:
            label_file.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")
            
            
def split_dataset(
    processed_images_dir,
    processed_labels_dir,
    train_ratio=0.7,
    eval_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Splits the dataset into train, eval, and test sets based on image IDs and copies them to respective directories.
    
    :param processed_images_dir: Directory containing the processed images.
    :param processed_labels_dir: Directory containing the processed label files.
    :param train_ratio: Proportion of data to be used for training.
    :param eval_ratio: Proportion of data to be used for evaluation.
    :param test_ratio: Proportion of data to be used for testing.
    :param random_state: Seed.
    """

    # List all image IDs based on image filenames
    image_files = [f for f in os.listdir(processed_images_dir) if f.endswith('.jpg')]
    image_ids = [os.path.splitext(f)[0] for f in image_files]

    print(f"Total images: {len(image_ids)}")

    # First split: Train and Temp (Eval + Test)
    train_ids, temp_ids = train_test_split(
        image_ids,
        test_size=(eval_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )

    eval_size_relative = eval_ratio / (eval_ratio + test_ratio)

    # Second split: Eval and Test
    eval_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - eval_size_relative),
        random_state=random_state,
        shuffle=True
    )

    print(f"Training set size: {len(train_ids)}")
    print(f"Evaluation set size: {len(eval_ids)}")
    print(f"Test set size: {len(test_ids)}")

    subsets = {
        'train': train_ids,
        'eval': eval_ids,
        'test': test_ids
    }

    for subset, ids in subsets.items():
        subset_images_dir = os.path.join(processed_images_dir, subset)
        subset_labels_dir = os.path.join(processed_labels_dir, subset)

        for image_id in ids:
            # Copy image
            src_image = os.path.join(processed_images_dir, f"{image_id}.jpg")
            dest_image = os.path.join(subset_images_dir, f"{image_id}.jpg")
            if os.path.exists(src_image):
                move(src_image, dest_image)
            else:
                print(f"Warning: Image {src_image} does not exist.")

            # Copy label
            src_label = os.path.join(processed_labels_dir, f"{image_id}.txt")
            dest_label = os.path.join(subset_labels_dir, f"{image_id}.txt")
            if os.path.exists(src_label):
                move(src_label, dest_label)
            else:
                print(f"Warning: Label file {src_label} does not exist.")

    print("Dataset split and organization completed.")



if __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(), 'data')
    INPUT_MAPPING = os.path.join(DATA_PATH, 'probe_labels.json')
    RAW_IMAGES = os.path.join(DATA_PATH, 'raw_images')
    PROCESSED_IMAGES = os.path.join(DATA_PATH, 'images')
    PROCESSED_LABELS = os.path.join(DATA_PATH, 'labels')

    # Convert labels and preprocess images
    convert_labels(
        input_json=INPUT_MAPPING,
        images_dir=RAW_IMAGES,
        output_dir_images=PROCESSED_IMAGES,
        output_dir_labels=PROCESSED_LABELS
    )

    # Split the dataset into train, eval, and test
    split_dataset(
        processed_images_dir=PROCESSED_IMAGES,
        processed_labels_dir=PROCESSED_LABELS,
        train_ratio=0.7,
        eval_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )