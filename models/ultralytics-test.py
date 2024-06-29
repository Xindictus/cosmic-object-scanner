import argparse
import cv2
import numpy as np
import os

# from sklearn.metrics import confusion_matrix, accuracy_score
from ultralytics import YOLO

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_CLASSES = 3


def get_category(category_id):
    return {
        0: "Galaxy",
        1: "Nebula",
        2: "Star Cluster"
    }[category_id]


def draw_boxes(image, predictions, threshold=0.5):
    for pred in predictions:
        bbox = pred[:4]  # x1, y1, x2, y2
        confidence = pred[4]

        if confidence >= threshold:
            class_id = int(pred[5])
            label = f'{get_category(class_id)}: {confidence:.2f}'

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Put label on bounding box
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

    return image


def plot_some_predictions(model):
    imgs_path = f'{CURRENT_DIR}/../data/yolo/visualize_predictions'
    predictions = model.predict(imgs_path)

    for prediction in predictions:
        # Get prediction
        boxes = prediction.boxes.data.cpu().numpy()
        path = prediction.path

        basename = os.path.basename(path)

        # Avoid re-creating bbox images for bboxes
        if 'bbox' not in basename:
            # Read the image
            image = cv2.imread(path)

            # Draw bounding boxes on the image
            image_with_boxes = draw_boxes(image, boxes)

            # Save the image with bounding boxes
            cv2.imwrite(f'{imgs_path}/bbox-{basename}', image_with_boxes)


# Function to match predictions and ground truth
def match_predictions(ground_truths, predictions, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for prediction in predictions:
        path = prediction.path
        basename = os.path.basename(path)

        matched_annotations = set()

        ground_truth = ground_truths[basename[:-4]]
        boxes = prediction.boxes.data.cpu().numpy()

        for box in boxes:
            pred_bbox = box[:4]
            pred_class = int(box[5])
            pred_matched = False

            for idx, gt in enumerate(ground_truth):
                gt_bbox = gt['bbox']
                gt_class = gt['class_id']

                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold and pred_class == gt_class and idx not in matched_annotations:
                    true_positives += 1
                    pred_matched = True
                    matched_annotations.add(idx)
                    break

            if not pred_matched:
                false_positives += 1

        false_negatives += len(ground_truth) - len(matched_annotations)

    return true_positives, false_positives, false_negatives


# Function to calculate accuracy
def calculate_accuracy(true_positives, false_positives, false_negatives):
    total_predictions = true_positives + false_positives
    total_annotations = true_positives + false_negatives

    if total_predictions == 0 or total_annotations == 0:
        return 0.0

    precision = true_positives / total_predictions
    recall = true_positives / total_annotations

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


# Function to create confusion matrix
def create_confusion_matrix(ground_truth, predictions):
    confusion_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

    for pred in predictions:
        pred_class = pred['class']
        pred_bbox = pred['bbox']
        pred_matched = False

        for gt in ground_truth:
            gt_class = gt['class_id']
            gt_bbox = gt['bbox']

            iou = calculate_iou(pred_bbox, gt_bbox)

            if iou >= iou_threshold and pred_class == gt_class:
                confusion_mat[gt_class, pred_class] += 1
                pred_matched = True
                break

        if not pred_matched:
            confusion_mat[0, pred_class] += 1  # False positive

    for gt in ground_truth:
        gt_class = gt['class_id']
        matched = False

        for pred in predictions:
            pred_class = pred['class']
            pred_bbox = pred['bbox']

            iou = calculate_iou(pred_bbox, gt_bbox)

            if iou >= iou_threshold and pred_class == gt_class:
                matched = True
                break

        if not matched:
            confusion_mat[gt_class, 0] += 1  # False negative

    return confusion_mat


# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # Calculate coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# Function to parse YOLO format label file
def parse_yolo_label(label_file, img_width, img_height):
    annotations = []

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        # Convert YOLO format to xmin, ymin, xmax, ymax format
        xmin = (x_center - width / 2) * img_width
        ymin = (y_center - height / 2) * img_height
        xmax = (x_center + width / 2) * img_width
        ymax = (y_center + height / 2) * img_height
        annotations.append({
            'class_id': int(class_id),
            'bbox': [xmin, ymin, xmax, ymax]
        })
    return annotations


def main(args):
    # Path to your pretrained model
    model_path = f'{CURRENT_DIR}/saved/{args.model}'
    test_img_path = f'{CURRENT_DIR}/../data/{args.test_img}'
    test_labels_path = f'{CURRENT_DIR}/../data/{args.test_lbl}'

    # Load the pretrained model
    model = YOLO(model_path)

    plot_some_predictions(model)

    # Perform inference on an image
    predictions = model.predict(test_img_path)

    # Dictionary to store ground truth annotations
    ground_truths = {}

    # Iterate over label files and parse annotations
    for label_file in os.listdir(test_labels_path):
        img_id, _ = os.path.splitext(label_file)
        label_path = os.path.join(test_labels_path, label_file)

        # Parse YOLO format label file
        ground_truths[img_id] = parse_yolo_label(label_path, 608, 608)

    # Define the IoU threshold for matching predictions to ground truth
    iou_threshold = 0.5

    # Initialize variables for metrics
    true_positives, false_positives, false_negatives = match_predictions(
        ground_truths,
        predictions,
        iou_threshold
    )

    print(true_positives, false_positives, false_negatives)

    # Calculate accuracy metrics
    precision, recall, f1_score = calculate_accuracy(
        true_positives,
        false_positives,
        false_negatives
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Create confusion matrix
    # confusion_mat = create_confusion_matrix(ground_truth, predictions)

    # print("Confusion Matrix:")
    # print(confusion_mat)


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description="YOLO Inference Example")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        required=True,
        help='Path to the pretrained YOLO model'
    )
    parser.add_argument(
        '-ti',
        '--test-img',
        type=str,
        required=True,
        help='Path to the test images dir for inference'
    )
    parser.add_argument(
        '-tl',
        '---test-lbl',
        type=str,
        required=True,
        help='Path to the test labels dir'
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args)
