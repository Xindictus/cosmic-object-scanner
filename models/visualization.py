import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


# Function to visualize the bounding boxes
def visualize_prediction(image, prediction, threshold=0.5):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    for box, score in zip(boxes, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.show()


def visualize_prediction_v2(
    image,
    prediction,
    threshold=0.5,
    category_names=None
):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
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
                edgecolor='r',
                facecolor='none',
            )
            ax.add_patch(rect)

            if category_names:
                label_name = category_names.get(label, f'Label {label}')
                ax.text(
                    xmin,
                    ymin - 10,
                    label_name,
                    color='red',
                    fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )

    plt.axis('off')
    plt.show()
