import json
import os
import shutil

from sklearn.model_selection import train_test_split


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = f'{CURRENT_DIR}/../data/coco'


def move_images(data, source_dir, target_dir):
    for img in data['images']:
        img_filename = img['file_name']
        shutil.copy(
            os.path.join(source_dir, img_filename),
            os.path.join(target_dir, img_filename)
        )


def filter_data(data, ids):
    filtered_data = {k: [] for k in data.keys()}
    image_ids = set(ids)

    for img in data['images']:
        if img['id'] in image_ids:
            filtered_data['images'].append(img)

    for ann in data['annotations']:
        if ann['image_id'] in image_ids:
            filtered_data['annotations'].append(ann)

    for key in ['categories', 'info']:
        filtered_data[key] = data[key]

    return filtered_data


def main():
    # Load the COCO dataset
    with open(f'{DATASET_DIR}/result.json', 'r') as f:
        coco_data = json.load(f)

    # Extract the image IDs
    image_ids = [img['id'] for img in coco_data['images']]

    # Split the data into train, validation, and test sets
    train_ids, test_ids = train_test_split(
        image_ids,
        test_size=0.20,
        random_state=42
    )

    train_data = filter_data(coco_data, train_ids)
    test_data = filter_data(coco_data, test_ids)

    # New dir paths
    train_dir = f'{DATASET_DIR}/train'
    test_dir = f'{DATASET_DIR}/test'

    # Create new dirs
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # annotations
    with open(f'{train_dir}/result.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{test_dir}/result.json', 'w') as f:
        json.dump(test_data, f)

    # Create new image dirs
    train_img_dir = f'{train_dir}/images'
    test_img_dir = f'{test_dir}/images'

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # Images copy
    move_images(train_data, f'{DATASET_DIR}', train_dir)
    move_images(test_data, f'{DATASET_DIR}', test_dir)


if __name__ == '__main__':
    main()
