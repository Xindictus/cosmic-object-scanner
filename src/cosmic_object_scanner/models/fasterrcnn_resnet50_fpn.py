import os
from itertools import cycle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, classification_report, roc_curve
from torchsummary import summary
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cosmic_object_scanner.models.coco_dataset import CocoDataset
from cosmic_object_scanner.models.engine import train_one_epoch

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_TRAIN_PATH = f"{CURRENT_DIR}/../data/coco/train/result.json"
ANNOTATIONS_TEST_PATH = f"{CURRENT_DIR}/../data/coco/test/result.json"
IMG_TRAIN_PATH = f"{CURRENT_DIR}/../data/coco/train"
IMG_TEST_PATH = f"{CURRENT_DIR}/../data/coco/test"
EPOCHS = 1
BATCH_SIZE = 2
MODEL_PATH = f"{CURRENT_DIR}/fasterrcnn_resnet50_fpn_{EPOCHS}_{BATCH_SIZE}.pth"
CLASS_NAMES = ["Galaxy", "Nebula", "Star Cluster"]


def calculate_iou(bb1: npt.NDArray[Any], bb2: npt.NDArray[Any]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bb1: Bounding box 1 with shape (4,). Contains [x1, y1, x2, y2] where
            (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        bb2: Bounding box 2 with shape (4,). Contains [x1, y1, x2, y2] where
            (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
        IoU score in the range [0, 1].

    Raises:
        AssertionError: If bounding box dimensions or coordinates are invalid.
    """
    assert bb1.shape == (4,) and bb2.shape == (4,)
    assert bb1[0] < bb1[2] and bb1[1] < bb1[3]
    assert bb2[0] < bb2[2] and bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou_value: float = float(intersection_area / float(bb1_area + bb2_area - intersection_area))
    assert iou_value >= 0.0
    assert iou_value <= 1.0
    return iou_value


def box_accuracy(
    preds: list[npt.NDArray[Any]],
    targets: list[npt.NDArray[Any]],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate bounding box accuracy based on IoU threshold.

    Args:
        preds: List of predicted bounding boxes, each as [x1, y1, x2, y2].
        targets: List of ground truth bounding boxes, each as [x1, y1, x2, y2].
        iou_threshold: IoU threshold for considering a match.

    Returns:
        Fraction of targets that have matching predictions (0.0 to 1.0).
    """
    correct = 0
    total = 0

    # Iterate over all target boxes
    for target_box in targets:
        # Iterate over all predicted boxes
        for pred_box in preds:
            iou = calculate_iou(pred_box, target_box)
            if iou >= iou_threshold:
                correct += 1
                break
        total += 1
    if total == 0:
        return 0.0
    return correct / total


def plot_learning_curve(train_loss: list[float], train_classifier_loss: list[float]) -> None:
    """Plot training loss and classifier loss curves.

    Args:
        train_loss: List of training loss values per iteration.
        train_classifier_loss: List of classifier loss values per iteration.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss)), train_loss, label="Training Loss", color="blue")
    plt.plot(
        range(len(train_classifier_loss)),
        train_classifier_loss,
        label="Classifier Training Loss",
        color="red",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()


def plot_roc_curve(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    y_score: npt.NDArray[Any],
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curves for each class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_score: Prediction scores/probabilities.
    """
    num_classes = 8
    all_labels_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # ROC Curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 7))
    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "red",
            "green",
            "yellow",
            "purple",
            "brown",
            "pink",
            "grey",
        ]
    )
    for i, color in zip(range(num_classes), colors, strict=False):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def print_classification_matrix(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> None:
    """Print classification report with per-class metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    """
    print("**************************************")
    # cc = CLASS_NAMES + ['background']
    cc = CLASS_NAMES
    class_report = classification_report(y_true, y_pred, target_names=cc)
    print("Classification Report:")
    print(class_report)
    print("**************************************")


def match_predictions_to_ground_truths(
    pred_boxes: list[npt.NDArray[Any]],
    true_boxes: list[npt.NDArray[Any]],
    pred_labels: list[int],
    true_labels: list[int],
    pred_scores: npt.NDArray[Any],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[float]]:
    """Match predicted boxes to ground truth boxes using IoU threshold.

    Args:
        pred_boxes: List of predicted bounding boxes.
        true_boxes: List of ground truth bounding boxes.
        pred_labels: List of predicted class labels.
        true_labels: List of ground truth class labels.
        pred_scores: Array of prediction confidence scores.
        iou_threshold: IoU threshold for considering a match.

    Returns:
        Tuple containing:
            - List of matched predicted labels
            - List of matched ground truth labels
            - List of matched scores
    """
    # Sort predictions by scores in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes_sorted: list[npt.NDArray[Any]] = [pred_boxes[int(i)] for i in sorted_indices]
    pred_labels_sorted: list[int] = [pred_labels[int(i)] for i in sorted_indices]
    pred_scores_sorted: list[float] = [float(pred_scores[int(i)]) for i in sorted_indices]

    matched_pred_labels: list[int] = []
    matched_true_labels: list[int] = []
    matched_scores: list[float] = []
    used_pred_indices: set[int] = set()

    for true_idx, true_box in enumerate(true_boxes):
        best_iou: float = 0
        best_pred_idx: int = -1
        for pred_idx, pred_box in enumerate(pred_boxes_sorted):
            if pred_idx in used_pred_indices:
                continue
            iou = calculate_iou(true_box, pred_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            used_pred_indices.add(best_pred_idx)
            matched_pred_labels.append(pred_labels_sorted[best_pred_idx])
            matched_scores.append(pred_scores_sorted[best_pred_idx])
            matched_true_labels.append(true_labels[true_idx])
        else:
            # Assuming 0 is the 'no object' class
            matched_pred_labels.append(0)
            # Assuming 0 is the score for 'no object'
            matched_scores.append(0.0)
            matched_true_labels.append(true_labels[true_idx])

    return matched_pred_labels, matched_true_labels, matched_scores


def collate_fn(batch: list[Any]) -> tuple[tuple[Any, ...], ...]:
    """Custom collate function  for DataLoader.

    Args:
        batch: List of samples from dataset.

    Returns:
        Tuple of (images, targets) unpacked from batch.
    """
    return tuple(zip(*batch, strict=False))


# Function to get the model
def get_model(num_classes: int) -> nn.Module:
    """Load and adapt Faster R-CNN ResNet50 FPN model for custom class count.

    Args:
        num_classes: Number of object classes (including background).

    Returns:
        Adapted Faster R-CNN model with custom classifier head.
    """
    # Load a model pre-trained on COCO
    model: nn.Module = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        # pretrained=True
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # type: ignore[union-attr]
    return model


def model_train(
    model: nn.Module,
    all_transforms: Any,
    device: torch.device,
) -> tuple[nn.Module, list[Any]]:
    """Train Faster R-CNN model on COCO dataset.

    Args:
        model: The model to train.
        all_transforms: Data augmentation transforms to apply.
        device: Device to train on (CPU or GPU).

    Returns:
        Tuple containing:
            - Trained model
            - List of training loss metrics per epoch
    """
    # dataset
    train_dataset = CocoDataset(IMG_TRAIN_PATH, ANNOTATIONS_TRAIN_PATH, all_transforms)
    train_data_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # parameters
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.009, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # print the model summary
    summary(model, input_size=(3, 608, 608))

    train_losses = []

    # training loop
    for epoch in range(EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_data_loader,
            device=device,
            epoch=epoch,
            print_freq=10,
        )
        train_losses.append(train_loss)
        # update the learning rate
        lr_scheduler.step()

    print("Training completed")

    # save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

    return model, train_losses


def test_model(
    model: nn.Module,
    data_loader_test: Any,
    device: torch.device,
    training_losses: list[Any],
) -> dict[str, Any]:
    """Evaluate trained model on test dataset and compute metrics.

    Args:
        model: The trained model to evaluate.
        data_loader_test: DataLoader providing test batches.
        device: Device to run evaluation on (CPU or GPU).
        training_losses: Training loss history for visualization.

    Returns:
        Dictionary containing test results and metrics.
    """
    all_preds = []
    all_labels = []
    all_scores = []
    all_boxes = []
    all_true_boxes = []

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs, strict=False):
                all_labels.append(target["labels"].cpu().numpy())
                all_preds.append(output["labels"].cpu().numpy())
                all_scores.append(output["scores"].cpu().numpy())
                all_boxes.append(output["boxes"].cpu().numpy())
                all_true_boxes.append(target["boxes"].cpu().numpy())

    y_true = []
    y_pred = []
    y_score = []

    for i in range(len(all_labels)):
        pred_boxes = all_boxes[i]
        true_boxes = all_true_boxes[i]
        pred_labels = all_preds[i]
        true_labels = all_labels[i]
        pred_scores = all_scores[i]

        y_pred_i, y_true_i, y_score_i = match_predictions_to_ground_truths(
            pred_boxes, true_boxes, pred_labels, true_labels, pred_scores
        )
        y_true.extend(y_true_i)
        y_pred.extend(y_pred_i)
        y_score.extend(y_score_i)

    train_loss = []
    train_classifier_loss = []
    for train_loss_dict in training_losses:
        train_loss.append(train_loss_dict.meters["loss"].avg)
        train_classifier_loss.append(train_loss_dict.meters["loss_classifier"].avg)

    box_accs = []

    for i in range(len(all_labels)):
        box_accs.append(box_accuracy(all_boxes[i], all_true_boxes[i]))

    box_acc = np.mean(box_accs)
    # Box Accuracy (using IoU)
    print("**************************************")
    print(f"\nBox Accuracy: {box_acc:.2f}\n")
    print("**************************************")

    plot_learning_curve(train_loss, train_classifier_loss)

    plot_roc_curve(np.array(y_true), np.array(y_pred), np.array(y_score))

    print_classification_matrix(np.array(y_true), np.array(y_pred))

    return {
        "box_accuracy": float(box_acc),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def main() -> None:
    # 3 classes (galaxy, nebula, star-cluster) + background
    num_classes = 4

    model = get_model(num_classes)

    # define the transformation
    all_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()]
    )

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the trained weights file exists
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        print("Model weights loaded. Skipping training.")
    else:
        model, train_losses = model_train(model, all_transforms, device)
        print(train_losses)

    test_dataset = CocoDataset(
        IMG_TEST_PATH, ANNOTATIONS_TEST_PATH, transforms.Compose([transforms.ToTensor()])
    )
    test_data_loader = utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    test_model(
        model=model, data_loader_test=test_data_loader, device=device, training_losses=train_losses
    )

    # model.eval()
    # Load the image
    # galaxies
    # img_path = f'{IMG_PATH}/images/00c1a7d2-1378.jpg'
    # img_path = f'{IMG_PATH}/images/4b2821f0-2205.jpg'
    # img_path = f'{IMG_PATH}/images/f41c7af9-2054.jpg'
    # img_path = f'{IMG_PATH}/images/5b381b61-2885.jpg'
    # star-cluster
    # img_path = f'{IMG_PATH}/images/efd97f0f-986.jpg'
    # img_path = f'{IMG_PATH}/images/fbf61c19-988.jpg'
    # nebula
    # img_path = f'{IMG_PATH}/images/f3b6d9c5-3616.jpg'
    # img_path = f'{IMG_PATH}/images/fe8ed0ef-957.jpg'
    # img_path = f'{IMG_PATH}/images/f461d039-4270.jpg'
    # img = Image.open(img_path).convert("RGB")

    # List all TEST files in the directory
    # files = os.listdir(f'{IMG_PATH}/images')

    # # Initialize an empty set to store unique values
    # unique_values = set()

    # for file in files:
    #     img_path = f'{IMG_PATH}/images/{file}'
    #     img = Image.open(img_path).convert("RGB")
    #     # Define the transformation
    #     img_tensor = all_transforms(img).unsqueeze(0).to(device)

    #     # Make the prediction
    #     with torch.no_grad():
    #         prediction = model(img_tensor)

    #         unique_values.update(prediction[0]['labels'])

    #         # if 0 in prediction[0]['labels']:
    #         #     print(file)
    #         #     break

    # print(unique_values)
    # Define the transformation
    # img_tensor = all_transforms(img).unsqueeze(0).to(device)

    # # Make the prediction
    # with torch.no_grad():
    #     prediction = model(img_tensor)

    # # Visualize the prediction
    # # visualize_prediction(img, prediction)
    # visualize_prediction_v2(
    #     image=img,
    #     prediction=prediction,
    #     threshold=0.6
    # )


if __name__ == "__main__":
    main()
