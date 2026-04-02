from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def get_category_names() -> dict[int, str]:
    """Get mapping of class indices to category names.

    Returns:
        Dictionary mapping class index to class name.
    """
    return {0: "Galaxy", 1: "Nebula", 2: "Star Cluster"}


# Function to visualize the bounding boxes
def visualize_ultralytics(
    image: Any,
    prediction: Any,
    threshold: float = 0.5,
    category_names: dict[int, str] | None = None,
) -> None:
    """Visualize predictions from Ultralytics YOLOv8 model.

    Args:
        image: Input image to visualize.
        prediction: Model predictions from ultralytics.
        threshold: Confidence threshold for displaying box_resorts.
        category_names: Optional dictionary mapping class indices to names.
    """
    if category_names is None:
        category_names = get_category_names()

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction[0].boxes.data.cpu().numpy()
    scores = boxes[:, 4]
    labels = boxes[:, 5].astype(int)
    boxes = boxes[:, :4]

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            if category_names:
                label_name = category_names.get(label, f"Label {label}")
                ax.text(
                    xmin,
                    ymin - 10,
                    label_name,
                    color="red",
                    fontsize=12,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    plt.axis("off")
    plt.show()


def visualize_prediction_v2(
    image: Any,
    prediction: Any,
    threshold: float = 0.5,
    category_names: dict[int, str] | None = None,
) -> None:
    """Visualize predictions from Faster R-CNN or similar PyTorch model.

    Args:
        image: Input image to visualize.
        prediction: Model predictions as dictionary containing 'boxes', 'scores', 'labels'.
        threshold: Confidence threshold for displaying boxes.
        category_names: Optional dictionary mapping class indices to names.
    """
    if category_names is None:
        category_names = get_category_names()
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction[0]["boxes"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()
    print(scores)
    # print(labels)

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            if category_names:
                label_name = category_names.get(label, f"Label {label}")
                ax.text(
                    xmin,
                    ymin - 10,
                    label_name,
                    color="red",
                    fontsize=12,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    plt.axis("off")
    plt.show()
