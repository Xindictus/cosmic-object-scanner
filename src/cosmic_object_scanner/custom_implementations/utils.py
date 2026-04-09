from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from cosmic_object_scanner.custom_implementations.constants import device


def iou(box1: Any, box2: Any, is_pred: bool = True) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Computes the IoU metric which measures the overlap between two bounding boxes.
    Can handle both coordinate-based (prediction) and dimension-based (label) formats.

    Args:
        box1: First bounding box in [x, y, width, height] format.
        box2: Second bounding box in [x, y, width, height] format.
        is_pred: If True, treats as prediction vs label comparison with sigmoid/exp
            transformations. If False, treats as width/height IoU calculation.
            Default: True

    Returns:
        torch.Tensor: IoU score(s) between 0 and 1 representing overlap ratio.

    Raises:
        ValueError: If bounding boxes are not in correct format.
    """
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = torch.abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = torch.abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score: torch.Tensor = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(
            box1[..., 1], box2[..., 1]
        )

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score_result: torch.Tensor = intersection_area / union_area

        # Return IoU score
        return iou_score_result


def nms(bboxes: list[Any], iou_threshold: float, threshold: float) -> list[Any]:
    """Apply Non-Maximum Suppression to remove redundant bounding boxes.

    Filters and removes overlapping bounding boxes, keeping only those with highest
    confidence scores and minimal overlap as determined by IoU threshold.

    Args:
        bboxes: List of bounding boxes in format [class_id, confidence, x1, y1, x2, y2].
        iou_threshold: Maximum IoU allowed between kept boxes. Boxes with IoU above
            this threshold to a kept box are removed. Range: [0, 1].
        threshold: Minimum confidence score to keep a bounding box. Boxes below
            this threshold are filtered out initially.

    Returns:
        list: Filtered list of bounding boxes after NMS, maintaining format
            [class_id, confidence, x1, y1, x2, y2].

    Note:
        - Bounding boxes are sorted by confidence in descending order before NMS
        - First box from sorted list is always kept
        - Overlapping boxes with lower confidence are removed
    """
    # Filter out bounding boxes with confidence below the threshold.
    bboxes = [box for box in bboxes if box[1] > threshold]
    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # Initialize the list of bounding boxes after non-maximum suppression.
    bboxes_nms = []

    while bboxes:
        # Get the first bounding box.
        first_box = bboxes.pop(0)

        # Iterate over the remaining bounding boxes.
        for box in bboxes:
            # If the bounding boxes do not overlap or if the first bounding box has
            # a higher confidence, then add the second bounding box to the list of
            # bounding boxes after non-maximum suppression.
            if (
                box[0] != first_box[0]
                or iou(
                    torch.tensor(first_box[2:]),
                    torch.tensor(box[2:]),
                )
                < iou_threshold
            ) and box not in bboxes_nms:
                # Add box to bboxes_nms
                bboxes_nms.append(box)

    # Return bounding boxes after non-maximum suppression.
    return bboxes_nms


def convert_cells_to_bboxes(
    predictions: Any, anchors: Any, s: int, is_predictions: bool = True
) -> list[Any]:
    """Convert cell predictions to bounding box coordinates.

    Converts YOLO-format cell predictions (with anchor boxes) into standard
    bounding box coordinates. Handles both model predictions and ground truth labels.

    Args:
        predictions: Model output tensor of shape [batch_size, num_anchors, grid, grid, 5+classes]
            where each cell contains [objectness, x, y, w, h, class_scores...].
        anchors: Anchor box templates as tensor of shape [num_anchors, 2] with [width, height].
        s: Grid size (e.g., 13, 26, 52 for YOLO multi-scale detection).
        is_predictions: If True, applies sigmoid to objectness/x/y and exp to w/h.
            If False, uses raw values. Default: True

    Returns:
        torch.Tensor: Bounding boxes of shape [batch_size, num_boxes, 6] where each box
            contains [class_id, objectness, x1, y1, x2, y2] in normalized coordinates.

    Note:
        - Predictions are converted from cell coordinates to image coordinates
        - Anchor boxes scale predictions to appropriate sizes
        - Grid coordinates are denormalized to [0, s] range
    """
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = predictions[..., 1:5]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Calculate cell indices
    cell_indices = (
        torch.arange(s).repeat(predictions.shape[0], 3, s, 1).unsqueeze(-1).to(predictions.device)
    )

    # Calculate x, y, width and height with proper scaling
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat((best_class, scores, x, y, width_height), dim=-1).reshape(
        batch_size, num_anchors * s * s, 6
    )

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()


def keep_prominent_boxes(boxes: list[Any], iou_threshold: float, count_threshold: int) -> list[Any]:
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array for easier manipulation
    boxes_array: Any = np.array(boxes)

    # Extract coordinates and scores
    x1: Any = boxes_array[:, 2] - boxes_array[:, 4] / 2
    y1: Any = boxes_array[:, 3] - boxes_array[:, 5] / 2
    x2: Any = boxes_array[:, 2] + boxes_array[:, 4] / 2
    y2: Any = boxes_array[:, 3] + boxes_array[:, 5] / 2
    labels: Any = boxes_array[:, 0]

    # Initialize list of kept boxes
    kept_boxes: list[Any] = []

    while len(boxes_array) > 0:
        # Pick the first box
        current_box = boxes_array[0]
        current_label = labels[0]

        current_x1 = x1[0]
        current_y1 = y1[0]
        current_x2 = x2[0]
        current_y2 = y2[0]

        # Calculate IoU of the picked box with the rest
        rest_x1 = x1[1:]
        rest_y1 = y1[1:]
        rest_x2 = x2[1:]
        rest_y2 = y2[1:]
        rest_labels = labels[1:]

        # Calculate the intersection areas
        inter_x1 = np.maximum(current_x1, rest_x1)
        inter_y1 = np.maximum(current_y1, rest_y1)
        inter_x2 = np.minimum(current_x2, rest_x2)
        inter_y2 = np.minimum(current_y2, rest_y2)

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        # Calculate the union areas
        current_area = (current_x2 - current_x1) * (current_y2 - current_y1)
        rest_area = (rest_x2 - rest_x1) * (rest_y2 - rest_y1)

        union_area = current_area + rest_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area

        # Find boxes with IoU greater than threshold and the same label
        similar_boxes_mask = (iou >= iou_threshold) & (rest_labels == current_label)
        similar_boxes_count = np.sum(similar_boxes_mask)

        if similar_boxes_count >= count_threshold:
            kept_boxes.append(current_box)

        # Remove the processed box and the similar boxes
        boxes_array = boxes_array[1:][~similar_boxes_mask]
        x1 = x1[1:][~similar_boxes_mask]
        y1 = y1[1:][~similar_boxes_mask]
        x2 = x2[1:][~similar_boxes_mask]
        y2 = y2[1:][~similar_boxes_mask]
        labels = labels[1:][~similar_boxes_mask]

    return kept_boxes


def plot_image(
    image: Any,
    boxes: list[Any],
    class_labels: list[str],
    iou_threshold: float = 0.5,
    count_threshold: int = 5,
) -> None:
    # Apply the prominent boxes logic to filter boxes
    filtered_boxes = keep_prominent_boxes(boxes, iou_threshold, count_threshold)

    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")
    # Getting 20 different colors from the color map for 20 different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    # Reading the image with OpenCV
    img = np.array(image)
    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Plotting the bounding boxes and labels over the image
    for box in filtered_boxes:
        # Get the class from the box
        class_pred = box[0]
        # Get the center x and y coordinates
        box = box[2:]
        # Get the upper left corner coordinates
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add class name to the patch
        plt.text(
            upper_left_x * w,
            upper_left_y * h,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Display the plot
    plt.axis("off")
    plt.show()


def save_checkpoint(model: Any, optimizer: Any, filename: str = "my_checkpoint.pth.tar") -> None:
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file: str, model: Any, optimizer: Any, lr: float) -> None:
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
