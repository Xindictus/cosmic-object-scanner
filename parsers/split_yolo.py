import os
import random
import shutil


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = f'{CURRENT_DIR}/../data/yolo'
IMG_DIR = os.path.join(DATASET_DIR, 'og', 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'og', 'labels')
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def main():
    # List all images assuming they have the same filename as labels but in 'images' directory
    image_files = os.listdir(IMG_DIR)
    random.shuffle(image_files)

    # Split ratios
    total_images = len(image_files)
    train_split = int(TRAIN_RATIO * total_images)
    val_split = int(VAL_RATIO * total_images)

    # Create directories if they don't exist
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(os.path.join(
            DATASET_DIR,
            'images',
            dir_path
        ), exist_ok=True)
        os.makedirs(os.path.join(
            DATASET_DIR,
            'labels',
            dir_path
        ), exist_ok=True)

    # Copy images and labels to train, val, test directories
    for i, image_file in enumerate(image_files):
        label_file = image_file.replace('.jpg', '.txt')

        src_image = os.path.join(IMG_DIR, image_file)
        src_label = os.path.join(LABELS_DIR, label_file)

        if i < train_split:
            shutil.copy(src_image, os.path.join(
                DATASET_DIR,
                'images',
                TRAIN_DIR,
                image_file
            ))
            shutil.copy(src_label, os.path.join(
                DATASET_DIR,
                'labels',
                TRAIN_DIR,
                label_file
            ))
        elif i < train_split + val_split:
            shutil.copy(src_image, os.path.join(
                DATASET_DIR,
                'images',
                VAL_DIR,
                image_file
            ))
            shutil.copy(src_label, os.path.join(
                DATASET_DIR,
                'labels',
                VAL_DIR,
                label_file
            ))
        else:
            shutil.copy(src_image, os.path.join(
                DATASET_DIR,
                'images',
                TEST_DIR,
                image_file
            ))
            shutil.copy(src_label, os.path.join(
                DATASET_DIR,
                'labels',
                TEST_DIR,
                label_file
            ))

    print("Dataset split into train, val, and test directories successfully.")


if __name__ == '__main__':
    main()
