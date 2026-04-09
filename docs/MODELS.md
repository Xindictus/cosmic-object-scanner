# Model Descriptions

## Overview

This project implements multiple object detection architectures using two frameworks (PyTorch and TensorFlow). Each model targets different use cases: accuracy, speed, and educational understanding.

## Models

### 1. Faster R-CNN ResNet50 FPN (PyTorch)

**File**: `src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py`

**Architecture**:

- Two-stage object detector
- ResNet50 backbone for feature extraction
- Feature Pyramid Network (FPN) for multi-scale detection
- Region Proposal Network (RPN) for proposal generation
- ROI pooling and classification/regression heads

**Framework**: PyTorch with torchvision pre-trained weights

**Key Parameters**:

```python
EPOCHS = 1             # Configurable
BATCH_SIZE = 2         # Limited by GPU memory
LEARNING_RATE = 0.01   # SGD with momentum
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3          # Learning rate step decay
GAMMA = 0.1            # LR decay factor
```

**Input/Output**:

- **Input**: RGB images (variable size, auto-padded)
- **Output**: Bounding boxes (x1, y1, x2, y2) + class labels + confidence scores

**Training**:

```bash
uv run python src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py
```

**Configuration** (edit in script):

- `ANNOTATIONS_TRAIN_PATH`: Path to COCO JSON annotations
- `IMG_TRAIN_PATH`: Path to training images
- `CLASS_NAMES`: Class list from dataset

**Performance** (10 epochs):

- Average Precision (AP): 62.4%
- AP@0.5 IoU: 85.1%
- AP@0.75 IoU: 68.3%
- Per-class AP: See [EVALUATION.md](EVALUATION.md)

**Strengths**:

- ✅ High accuracy (two-stage detector)
- ✅ Pre-trained backbone for transfer learning
- ✅ Handles variable input sizes
- ✅ Well-documented architecture

**Limitations**:

- ❌ Slower inference speed than single-stage detectors
- ❌ Higher computational cost
- ❌ Requires ~8GB GPU memory

### 2. YOLOv3 from Scratch (TensorFlow)

**File**: `src/cosmic_object_scanner/custom_implementations/`

**Architecture**:

- Single-stage detector
- Darknet-53 backbone (custom implementation)
- Three detection heads at different scales (13×13, 26×26, 52×52)
- Anchor boxes (9 default)
- Multi-scale feature pyramid

**Framework**: TensorFlow (Keras)

**Key Components**:

- `building_blocks.py` — Conv blocks (`CNNBlock`), residual connections (`ResidualBlock`), scale predictions (`ScalePrediction`)
- `constants.py` — Anchor configurations, grid settings, device selection
- `data_utils.py` — Batch loading and preprocessing
- `model_utils.py` — Training utilities (`YOLOLoss`)
- `utils.py` — IoU, NMS, bounding box conversion

**Key Parameters**:

```python
GRID_SIZES = [13, 26, 52]     # Multi-scale detection
ANCHORS = [...]               # 9 anchor boxes
BATCH_SIZE = 16               # Configurable
LEARNING_RATE = 0.001
EPOCHS = 20                   # Standard training
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
```

**Training**:

```bash
uv run python src/cosmic_object_scanner/custom_implementations/yolo/model.py
```

**Performance** (20 epochs):

- Mean Average Precision (mAP): 58.7%
- Precision: 89.2%
- Recall: 71.3%
- Per-class metrics: See [EVALUATION.md](EVALUATION.md)

**Strengths**:

- ✅ Fast inference (real-time capable)
- ✅ Educational value (built from scratch)
- ✅ Lower memory footprint
- ✅ Multi-scale detection

**Limitations**:

- ❌ Slightly lower accuracy than Faster R-CNN
- ❌ No pre-training available
- ❌ Training from scratch is slower

### 3. Ultralytics YOLOv8 (PyTorch)

**File**: `src/cosmic_object_scanner/models/ultralytics_yolo.py`

**Architecture**:

- Modern YOLOv8 architecture
- Neck: PANet (Path Aggregation Network)
- Head: Decoupled detection head
- Latest optimizations and improvements

**Framework**: PyTorch (via Ultralytics library)

**Key Parameters**:

```python
from cosmic_object_scanner.models.ultralytics_yolo import train_yolov8

# Train with defaults
results = train_yolov8(
    model_name='yolov8m.pt',
    data_config='data.yaml',
    epochs=100,
    image_size=640,
)
```

**Usage**:

```bash
uv run python src/cosmic_object_scanner/models/ultralytics_yolo.py
```

**Strengths**:

- ✅ State-of-the-art accuracy
- ✅ Production-ready (fully optimized)
- ✅ Easy API (minimal code)
- ✅ Built-in model export

**Limitations**:

- ❌ Less educational (black-box library)
- ❌ High dependency on Ultralytics library

### 4. Hybrid TensorFlow Model

**Files**: `src/cosmic_object_scanner/custom_implementations/`

**Architecture**:
Two-component system:

1. **Classifier** (`classifier.py`)
   - Multi-class classification network via `build_sequential_classifier()` and `build_classifier()` factory functions
   - Input: Cropped image regions
   - Output: Class probabilities

2. **BBox Regressor** (`bbox_regressor.py`)
   - Continuous bounding box adjustment via `build_regressor()` factory function
   - Input: Anchors or initial proposals
   - Output: Regressed box coordinates (4 values)

3. **Combined Model** (`model.py`)
   - Feature extraction → adaptor → dual heads (classifier + regressor)
   - Via `build_model()` factory function

**Framework**: TensorFlow (Keras)

**Components**:

```python
from cosmic_object_scanner.custom_implementations.classifier import (
    build_sequential_classifier,
    build_classifier,
)
from cosmic_object_scanner.custom_implementations.bbox_regressor import build_regressor
from cosmic_object_scanner.custom_implementations.model import build_model

# Sequential classifier
classifier_model = build_sequential_classifier(num_classes=3)

# Functional classifier
classifier_model = build_classifier(num_classes=3)

# Regressor
regressor_model = build_regressor()

# Combined model with dual heads
full_model = build_model(num_classes=3)
```

**Strengths**:

- ✅ Modular design (separate classifier and regressor)
- ✅ Educational for understanding component architecture
- ✅ Efficient training of independent components

**Limitations**:

- ❌ Requires good anchor/proposal generation
- ❌ Performance depends on proposal quality

## Model Comparison

| Metric | Faster R-CNN | YOLOv3 | YOLOv8 | Hybrid |
|--------|--------------|--------|--------|--------|
| **Framework** | PyTorch | TensorFlow | PyTorch | TensorFlow |
| **Type** | Two-stage | Single-stage | Single-stage | Components |
| **Accuracy (AP)** | 62.4% | 58.7% | ~65%* | ~50%* |
| **Speed (fps)** | 20-30 | 50-100 | 80-120 | 30-50 |
| **Memory (GB)** | 8 | 6 | 7 | 4 |
| **Training Time** | 4-6 hrs | 3-5 hrs | 2-4 hrs | 1-2 hrs |
| **Pre-trained** | ✅ | ❌ | ✅ | ❌ |
| **Educational** | Medium | High | Low | High |
| **Production-Ready** | ✅ | ✅ | ✅✅ | ⚠️ |

*Estimated from architecture; requires full training validation

## Training Procedures

### General Training Loop

All models follow similar patterns:

```python
from cosmic_object_scanner.models import engine

for epoch in range(num_epochs):
    # Training phase
    train_metrics = engine.train_one_epoch(model, train_loader, optimizer, device)

    # Validation phase
    val_metrics = engine.evaluate(model, val_loader, device)

    # Logging
    print(f"Epoch {epoch}: AP={val_metrics['ap']:.4f}")

    # Checkpoint
    if val_metrics['ap'] > best_ap:
        save_checkpoint(model, epoch)

    # Learning rate decay
    scheduler.step()
```

### Faster R-CNN Training

```bash
# Setup
uv run python -c "from cosmic_object_scanner.models import fasterrcnn_resnet50_fpn"

# Edit configuration in fasterrcnn_resnet50_fpn.py:
# - EPOCHS, BATCH_SIZE
# - ANNOTATIONS_TRAIN_PATH, IMG_TRAIN_PATH
# - CLASS_NAMES

# Train
uv run python src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py

# Output: fasterrcnn_resnet50_fpn_<epochs>_<batch_size>.pth
```

### YOLOv3 Training

```bash
# Prepare data in YOLO format (see data/split_yolo.py)
uv run python src/cosmic_object_scanner/data/split_yolo.py --input data/coco --output data/yolo

# Train
uv run python src/cosmic_object_scanner/custom_implementations/yolo/model.py

# Configuration in model.py or via command line
```

### YOLOv8 Training

```bash
# Prepare data.yaml
cat > data.yaml << EOF
path: /path/to/data
train: images/train
val: images/val
test: images/test

nc: 3
names: ['Galaxy', 'Nebula', 'Star Cluster']
EOF

# Train
uv run python src/cosmic_object_scanner/models/ultralytics_yolo.py
```

## Inference

### Faster R-CNN

```python
from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import get_model
from cosmic_object_scanner.models.visualization import visualize_prediction_v2
import torch
from PIL import Image

# Load model
model = get_model(num_classes=4)  # +1 for background
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Predict
image = Image.open('test.jpg')
# ... preprocess ...
predictions = model([image_tensor])

# Visualize
visualize_prediction_v2(image, predictions)
```

### YOLOv8

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.5, iou=0.5)
```

## Model Selection Guide

**Use Faster R-CNN when**:

- Maximum accuracy is critical
- Inference speed is not a bottleneck
- You have GPU memory and training time
- Transfer learning is desired

**Use YOLOv3 when**:

- Real-time inference is needed
- You want to understand architecture details
- GPU memory is limited
- You're doing research/experimentation

**Use YOLOv8 when**:

- You need production-ready solution
- You want latest improvements
- You prefer simple API
- Model deployment is important

**Use Hybrid when**:

- Understanding component-based architectures
- Fine-tuning classification vs. localization separately
- Research on proposal-based detection

## Performance Tuning

### Improving Accuracy

1. **Data Augmentation**: Enhance transforms in `transforms.py`
2. **Longer Training**: Increase epochs
3. **Lower Learning Rate**: Slower but steadier convergence
4. **Larger Batch Size**: Better gradient estimates (if memory allows)

### Improving Speed

1. **Model Quantization**: Convert to INT8
2. **Smaller Models**: Use YOLOv8n (nano) instead of m (medium)
3. **Batch Inference**: Process multiple images
4. **Remove Post-Processing**: If NMS not needed

### Memory Optimization

1. **Reduce Batch Size**
2. **Use Gradient Checkpointing**: Trade compute for memory
3. **FP16 Mixed Precision**: `torch.cuda.amp`
4. **Model Pruning**: Remove unnecessary parameters

## References

- [Faster R-CNN](https://arxiv.org/abs/1506.01497) - Ren et al., 2015
- [YOLOv3](https://arxiv.org/abs/1804.02767) - Redmon & Farhadi, 2018
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) - Lin et al., 2017
- [COCO Dataset](https://cocodataset.org/)
