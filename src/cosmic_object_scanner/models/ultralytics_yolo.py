"""YOLOv8 training wrapper using the Ultralytics library."""

from typing import Any

from ultralytics import YOLO


def train_yolov8(
    model_name: str = "yolov8n.pt",
    data_config: str = "data.yaml",
    epochs: int = 2,
    image_size: int = 608,
) -> Any:
    """Train a YOLOv8 model.

    Args:
        model_name: Pretrained model to load.
        data_config: Path to the YOLO data configuration YAML.
        epochs: Number of training epochs.
        image_size: Input image size.

    Returns:
        Training results from Ultralytics.
    """
    model = YOLO(model_name)
    results: Any = model.train(data=data_config, epochs=epochs, imgsz=image_size)
    return results


if __name__ == "__main__":
    train_yolov8()
