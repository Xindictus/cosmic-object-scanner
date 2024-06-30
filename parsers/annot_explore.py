import json
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import Counter


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = f'{CURRENT_DIR}/../data/coco'


def main():
    # Load the COCO result.json file
    with open(f'{DATASET_DIR}/result.json', 'r') as f:
        coco_data = json.load(f)

    # Extract the number of annotations and images
    num_annotations = len(coco_data['annotations'])
    num_images = len(coco_data['images'])

    print(f'Total number of annotations: {num_annotations}')
    print(f'Total number of images: {num_images}')

    # Extract the breakdown of classes
    category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}
    class_counts = Counter([category_id_to_name[ann['category_id']] for ann in coco_data['annotations']])

    print('Class breakdown:')
    for class_name, count in class_counts.items():
        print(f'{class_name}: {count}')

    # Plot the breakdown of classes
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
    plt.bar(
        class_counts.keys(),
        class_counts.values(),
        color=colors
    )
    plt.xlabel('Class')
    plt.ylabel('Number of annotations')
    plt.title('Breakdown of Classes in Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.show()


if __name__ == '__main__':
    main()
