# Dataset Documentation

## Overview

The Cosmic Object Scanner dataset contains 5,000+ astronomical images manually annotated with bounding boxes for three classes of stellar objects: galaxies, nebulae, and star clusters. Annotations were created using Label Studio and are provided in COCO format.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 5,000+ |
| **Annotated Objects** | ~15,000 |
| **Classes** | 3 (Galaxy, Nebula, Star Cluster) |
| **Annotation Tool** | Label Studio |
| **Format** | COCO JSON + YOLO TXT (converted) |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Resolution** | Variable (see Distribution below) |
| **Color Space** | RGB |

## Classes

### 1. Galaxy

**Description**: Large gravitational structures containing billions of stars

**Characteristics**:

- Large spatial extent (typically 50-500 pixels)
- Recognizable morphological patterns (spiral, elliptical, irregular)
- Well-defined edges with stellar halo
- Multiple sub-components (bulge, disk, arms)

**Count**: ~1,800 instances
**Avg. Size**: 200×200 pixels
**Examples**: Andromeda-like spirals, elliptical galaxies

### 2. Nebula

**Description**: Interstellar clouds of gas and dust

**Characteristics**:

- Variable morphology (emission, reflection, dark)
- Diffuse boundaries
- Often associated with stellar formation
- Can appear isolated or near stars/clusters

**Count**: ~1,600 instances
**Avg. Size**: 100×100 pixels
**Types**:

- Emission nebulae (red, hydrogen-rich)
- Reflection nebulae (bluish)
- Dark nebulae (absorption)

### 3. Star Cluster

**Description**: Groups of gravitationally bound stars

**Characteristics**:

- Dense point-like structures
- Well-defined boundaries
- Multiple stars in compact region
- May have tidal disruption

**Count**: ~1,600 instances
**Avg. Size**: 80×80 pixels
**Types**:

- Open clusters (loose configuration)
- Globular clusters (spherical, dense)

## Bounding Box Statistics

### Size Distribution

```
Galaxy:       50th percentile = 150px,  95th percentile = 450px
Nebula:       50th percentile = 80px,   95th percentile = 300px
Star Cluster: 50th percentile = 60px,   95th percentile = 200px
```

### Aspect Ratio

```
Galaxy:       mean = 0.95 (nearly square)
Nebula:       mean = 1.1 (slightly wider)
Star Cluster: mean = 0.98 (nearly square)
```

### Density (objects per image)

```
Average: 3.0 objects per image
Median:  2.0 objects per image
Min:     1 object
Max:     15 objects
```

## Data Format

### COCO Format

**Location**: `data/coco/`

**Structure**:
```
data/coco/
├── train/
│   ├── images/           # Training images (JPEG)
│   └── result.json       # COCO annotations
├── test/
│   ├── images/           # Test images (JPEG)
│   └── result.json       # COCO annotations
└── val/                  # Validation set (if used)
```

**COCO JSON Format** (`result.json`):
```json
{
  "images": [
    {"id": 0, "file_name": "image_001.jpg", "height": 512, "width": 512}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],  # COCO format
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "Galaxy"},
    {"id": 2, "name": "Nebula"},
    {"id": 3, "name": "Star Cluster"}
  ]
}
```

### YOLO Format

**Location**: `data/yolo/`

**Structure**:
```
data/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**YOLO TXT Format** (one per image):
```
<class_id> <x_center> <y_center> <width> <height>  # normalized (0-1)
<class_id> <x_center> <y_center> <width> <height>
...
```

**data.yaml**:
```yaml
path: /path/to/data/yolo
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: Galaxy
  1: Nebula
  2: Star Cluster
```

### Label Studio Export

**Original Format**: Label Studio project exports with task definitions, annotations, and metadata.

**Conversion**: See `split_coco.py` and `split_yolo.py` for conversion utilities.

## Data Loading

### COCO Format (PyTorch)

```python
from cosmic_object_scanner.models.coco_dataset import CocoDataset
from torch.utils.data import DataLoader

# Load COCO dataset
dataset = CocoDataset(
    root='data/coco/train',
    annFile='data/coco/train/result.json',
    transforms=None
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# Iterate
for images, targets in dataloader:
    # images: list of tensors
    # targets: list of dicts with 'boxes', 'labels'
    pass
```

### YOLO Format (TensorFlow)

```python
from cosmic_object_scanner.custom_implementations.yolo.Data_utils import YoloDataLoader

# Load YOLO dataset
train_loader = YoloDataLoader(
    images_dir='data/yolo/images/train',
    labels_dir='data/yolo/labels/train',
    batch_size=16,
    img_size=416
)

# Iterate
for images, targets in train_loader:
    # images: batch of shape (batch, 416, 416, 3)
    # targets: batch of shape (batch, max_objects, 5)
    pass
```

## Data Augmentation

### Albumentations Pipeline (COCO)

```python
from cosmic_object_scanner.models.transforms import get_transform

# Training transforms
train_transforms = get_transform(train=True)

# Validation transforms
val_transforms = get_transform(train=False)

# Applied transforms
# - HorizontalFlip (p=0.5)
# - VerticalFlip (p=0.1)
# - RandomBrightnessContrast (p=0.2)
# - RandomRotate90 (p=0.1)
# - GaussNoise (p=0.1)
# - MotionBlur (p=0.1)
```

### Custom Augmentation (YOLO)

Implemented in `Data_utils.py`:

- Random horizontal/vertical flips
- Random brightness/contrast adjustment
- Random rotation
- Random scale
- Mosaic augmentation

## Data Splits

### Official Split (70/15/15)

```shell
Train: 3,500 images (~10,500 objects)
Val:   750 images (~2,250 objects)
Test:  750 images (~2,250 objects)
```

### Stratified Split

Objects per class balanced across splits:

| Split | Galaxy | Nebula | Star Cluster | Total |
|-------|--------|--------|--------------|-------|
| Train | 1,260 | 1,140 | 1,140 | 3,540 |
| Val | 270 | 245 | 245 | 760 |
| Test | 270 | 245 | 245 | 760 |

## Dataset Preparation

### Converting from Label Studio

```bash
# Export from Label Studio as COCO JSON

# Convert to YOLO format
uv run python src/cosmic_object_scanner/data/split_yolo.py \
    --input data/label_studio_export.json \
    --output data/yolo

# Or split existing COCO
uv run python src/cosmic_object_scanner/data/split_coco.py \
    --input data/coco_original \
    --output data/coco \
    --train-ratio 0.7
```

### Exploring Annotations

```bash
# Analyze dataset
uv run python src/cosmic_object_scanner/data/annot_explore.py

# Output:
# - Class distribution
# - Image statistics (size, aspect ratio)
# - Annotation statistics (objects per image)
# - Visualization of sample annotations
```

## Quality Notes

### Data Quality Checks

1. **Annotation Coverage**: All visible objects annotated
2. **Boundary Accuracy**: Boxes tightly fit objects
3. **Label Correctness**: Manually verified
4. **No Duplicates**: Dataset cleaned of duplicated images

### Known Issues

- **Large Objects**: Some large galaxies may be cropped at image boundaries
- **Overlapping Objects**: Some objects overlap; boxes cover visible extent
- **Resolution Variation**: Images have different resolutions (see Variable Resolution note)

### Variable Resolution

Images were not resized to a standard resolution to preserve original details. Models handle this via:

- Auto-padding (for Faster R-CNN)
- Resizing with aspect ratio preservation (for YOLO)
- Dynamic batch creation (both frameworks)

**Resolution Range**:

- Min: 256×256 pixels
- Max: 2048×2048 pixels
- Mean: ~800×800 pixels

## Dataset Usage Examples

### Data Exploration

```python
from cosmic_object_scanner.models.coco_dataset import CocoDataset
import matplotlib.pyplot as plt

dataset = CocoDataset('data/coco/train', 'data/coco/train/result.json')

# Get sample
image, target = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Boxes: {target['boxes']}")
print(f"Labels: {target['labels']}")

# Visualize
plt.imshow(image)
for box in target['boxes']:
    x1, y1, x2, y2 = box
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='r'))
plt.show()
```

### Training with Dataset

```bash
# Edit model script to point to data location
# models/fasterrcnn_resnet50_fpn.py

# Then run training
uv run python src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py
```

## References

- [COCO Dataset](https://cocodataset.org/)
- [YOLO Format](https://docs.ultralytics.com/datasets/detect/)
- [Label Studio](https://labelstud.io/guide/)
- [Albumentations](https://albumentations.ai/)
