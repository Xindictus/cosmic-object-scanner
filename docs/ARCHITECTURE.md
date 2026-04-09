# Architecture Guide

## Project Structure

```
cosmic-object-scanner/
├── src/cosmic_object_scanner/          # Main package
│   ├── __init__.py
│   ├── py.typed                        # Type hints marker
│   ├── models/                         # PyTorch-based models
│   │   ├── __init__.py
│   │   ├── fasterrcnn_resnet50_fpn.py # Main Faster R-CNN implementation
│   │   ├── engine.py                  # Training loop
│   │   ├── coco_dataset.py            # COCO dataset loader
│   │   ├── coco_eval.py               # COCO evaluation metrics
│   │   ├── coco_utils.py              # COCO utilities
│   │   ├── transforms.py             # Data augmentation
│   │   ├── visualization.py           # Result visualization
│   │   ├── utils.py                   # Utility functions
│   │   ├── ultralytics_test.py        # Ultralytics YOLO testing
│   │   └── ultralytics_yolo.py        # Ultralytics YOLO training
│   │
│   ├── custom_implementations/         # TensorFlow & scratch implementations
│   │   ├── __init__.py
│   │   ├── building_blocks.py         # YOLOv3 building blocks (conv, residual)
│   │   ├── model.py                   # TF hybrid model (classifier + regressor)
│   │   ├── classifier.py             # Multi-class CNN classifier
│   │   ├── bbox_regressor.py          # Bounding box regressor
│   │   ├── constants.py               # Model constants & hyperparameters
│   │   ├── utils.py                   # YOLOv3 utilities (IoU, NMS, etc.)
│   │   ├── data_utils.py             # Data loading & preprocessing
│   │   └── model_utils.py            # Training utilities (YOLOLoss)
│   │
│   └── data/                           # Data utilities
│       ├── __init__.py
│       ├── split_coco.py              # COCO format splitting
│       ├── split_yolo.py              # YOLO format splitting
│       └── annot_explore.py           # Annotation exploration
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration & fixtures
│   ├── custom_implementations/        # TF & scratch model tests
│   │   ├── test_building_blocks.py
│   │   ├── test_constants.py
│   │   ├── test_data_utils.py
│   │   ├── test_model_utils.py
│   │   ├── test_tf_models.py
│   │   └── test_utils.py
│   ├── models/                        # PyTorch model tests
│   │   ├── test_coco_eval.py
│   │   ├── test_coco_utils.py
│   │   ├── test_fasterrcnn.py
│   │   ├── test_imports.py
│   │   ├── test_model_validation.py
│   │   ├── test_models_utils.py
│   │   ├── test_transforms.py
│   │   ├── test_ultralytics.py
│   │   ├── test_utils.py
│   │   └── test_visualization.py
│   └── data/                          # Data utility tests
│       └── test_imports.py
│
├── docs/                               # Documentation
│   ├── ARCHITECTURE.md                # This file
│   ├── MODELS.md                      # Model descriptions
│   ├── DATASET.md                     # Dataset information
│   ├── EVALUATION.md                  # Evaluation results
│   └── DEV_GUIDE.md                   # Development guide
│
├── experimental/                       # Experimental code
│   ├── dataset.ipynb                  # Dataset exploration
│   ├── train.csv                      # Training data
│   └── test.csv                       # Test data
│
├── data/                               # Dataset directory (not in repo)
│   ├── yolo/                          # YOLO format dataset
│   └── coco/                          # COCO format dataset
│
├── pyproject.toml                      # Project config + tool settings
├── uv.lock                             # Dependency lock file
├── .pre-commit-config.yaml             # Pre-commit hooks
├── .gitignore                          # Git ignore rules
├── .github/
│   └── workflows/
│       ├── lint-and-test.yml          # CI: Lint, type check, test
│       └── model-validation.yml       # CI: Model validation (nightly)
├── README.md                           # Project overview
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                             # MIT License
├── Report.pdf                          # Technical report
└── Presentation.pptx                   # Project presentation
```

## Module Organization

### models/

**Primary focus**: PyTorch-based object detection models

**Key components**:
- **fasterrcnn_resnet50_fpn.py** — Main Faster R-CNN implementation with training and evaluation
- **engine.py** — Training loop (`train_one_epoch`, `evaluate`)
- **coco_dataset.py** — COCO dataset loader for PyTorch
- **coco_eval.py** — Evaluation using COCOEval metrics
- **transforms.py** — Data augmentation pipeline (Albumentations integration)
- **visualization.py** — Visualization utilities for predictions
- **ultralytics_yolo.py** — Ultralytics YOLO training via `train_yolov8()` factory function
- **ultralytics_test.py** — Ultralytics YOLO evaluation and testing

**Dependencies**: torch, torchvision, pycocotools, albumentations

### custom_implementations/

**Primary focus**: Models built from scratch or with alternative frameworks (TensorFlow/Keras)

All files are in a flat directory structure (no subdirectories):

**YOLOv3 from scratch**:
- **building_blocks.py** — Convolutional blocks (`CNNBlock`), residual connections (`ResidualBlock`), scale predictions (`ScalePrediction`)
- **constants.py** — Model configuration (anchor boxes, grid sizes, device selection)
- **utils.py** — Model-specific utilities (IoU, NMS, bounding box conversion)
- **data_utils.py** — Data loading and preprocessing
- **model_utils.py** — Training utilities (`YOLOLoss`)

**TensorFlow hybrid model**:
- **classifier.py** — Multi-class CNN classifier via `build_sequential_classifier()` and `build_classifier()` factory functions
- **bbox_regressor.py** — Bounding box regression via `build_regressor()` factory function
- **model.py** — Combined classifier + regressor via `build_model()` with feature extraction, adaptor, and dual heads

### data/

**Primary focus**: Dataset utilities and data processing

**Key components**:
- **split_coco.py** — Convert and split datasets to COCO format
- **split_yolo.py** — Convert and split datasets to YOLO format
- **annot_explore.py** — Explore and visualize annotations

**Dependencies**: pycocotools, pandas

## Design Decisions

### Dual Framework Support (PyTorch + TensorFlow)

**Rationale**:
- **PyTorch**: Faster R-CNN implementation with modern tooling (torchvision pre-trained models)
- **TensorFlow**: Custom YOLOv3 and hybrid model implementations for educational purposes
- Reflects real-world multi-framework environments

**Trade-offs**:
- ✅ Demonstrates multiple frameworks
- ✅ Realistic production scenario
- ❌ Increased complexity and dependencies

### Flat Module Layout (custom_implementations/)

**Rationale**:
- All custom implementation files share constants and utilities
- Eliminates unnecessary nesting and import complexity
- Simpler packaging and testing

**Pattern**:
```python
# Import from custom implementations
from cosmic_object_scanner.custom_implementations.classifier import build_sequential_classifier
from cosmic_object_scanner.custom_implementations.bbox_regressor import build_regressor
from cosmic_object_scanner.custom_implementations.model import build_model

# Import from models
from cosmic_object_scanner.models.coco_eval import CocoEvaluator
from cosmic_object_scanner.models.ultralytics_yolo import train_yolov8
```

### Factory Functions Over Module-Level Side Effects

**Rationale**:
- Model construction is deferred until explicitly called
- Importing a module never triggers GPU allocation or model instantiation
- Enables safe testing and static analysis

**Pattern**:
```python
# Good: Factory function
model = build_sequential_classifier(num_classes=3)

# Good: Guarded entry point
if __name__ == "__main__":
    train_yolov8()
```

### Data Format Support (COCO + YOLO)

**Rationale**:
- COCO format: Standard for object detection with rich metadata
- YOLO format: Lightweight, efficient for training
- Conversion utilities allow experiment with both

### Type Hints & Mypy Strict Mode

**Rationale**:
- Catch errors at development time
- Improve code readability and maintainability
- Enable IDE autocompletion
- Production-ready quality signal

**Implementation**:
- All public function signatures have type hints
- `mypy src/ --strict` passes with 0 errors
- `py.typed` marker for PEP 561 compliance

## Dependency Management

### Frameworks

| Package | Version | Purpose | Module |
|---------|---------|---------|--------|
| torch | ≥2.0.0 | PyTorch core | models/ |
| torchvision | ≥0.17.0 | PyTorch vision (pre-trained, transforms) | models/ |
| tensorflow | ≥2.13.0 | TensorFlow core | custom_implementations/ |
| ultralytics | ≥8.0.0 | YOLO implementations | models/ |

### Computer Vision

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Image processing |
| pillow | ≥10.0.0 | Image I/O |
| albumentations | ≥1.3.0 | Data augmentation |

### Evaluation & Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| pycocotools | ≥2.0.7 | COCO metrics |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Statistical visualization |

### Development

| Package | Version | Purpose | Tool |
|---------|---------|---------|------|
| ruff | ≥0.1.0 | Linting | ruff check . |
| black | ≥23.0.0 | Formatting | black . |
| isort | ≥5.12.0 | Import sorting | isort . |
| mypy | ≥1.7.0 | Type checking | mypy src/ --strict |
| pytest | ≥7.4.0 | Testing | pytest tests/ |
| pytest-cov | ≥4.1.0 | Coverage | pytest --cov |
| pre-commit | ≥3.5.0 | Git hooks | pre-commit install |

## Development Workflow

### Local Development

```bash
# 1. Setup environment
uv sync --dev

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and write tests
# Edit src/, tests/

# 4. Run linters and tests
uv run ruff check src/ tests/
uv run black src/ tests/
uv run mypy src/ --strict
uv run pytest tests/ --cov

# 5. Install pre-commit hooks
uv run pre-commit install

# 6. Commit changes
git add .
git commit -m "feat: Add feature description"

# 7. Push and create PR
git push origin feature/my-feature
```

### Code Quality Gates

**Local** (pre-commit hooks):
- ruff linting
- black formatting check
- mypy type checking

**GitHub Actions** (on PR):
- Lint, format, type checks
- Test suite with coverage
- Model validation (optional)

## Performance Considerations

### Model Training

**Memory Usage**:
- Faster R-CNN: ~8GB GPU memory (batch_size=2)
- YOLOv3: ~6GB GPU memory
- Hybrid model: ~4GB GPU memory

**Training Time** (per epoch on single GPU):
- Faster R-CNN: ~15-20 minutes
- YOLOv3: ~10-15 minutes
- Full dataset with 5000+ images

### Inference

**Typical Throughput** (on single GPU):
- Faster R-CNN: ~20-30 images/sec
- YOLOv3: ~50-100 images/sec
- CPU inference: 1-5 images/sec

## Testing Strategy

### Unit Tests

- **Custom implementations**: YOLOv3 utilities (IoU, NMS), building blocks, constants, TF model construction
- **Models**: COCO evaluation, Faster R-CNN helpers, transforms, visualization
- **Data loading**: COCO/YOLO dataset loaders, import validation

### Integration Tests

- **Training loop**: Small 10-image subset
- **Evaluation**: Metric computation

### Not in CI (Local/Manual)

- Full model training (15-30 min per model)
- Inference on full 5000+ image dataset

## Adding New Components

### Adding a New Model

1. Create `src/cosmic_object_scanner/models/new_model.py`
2. Implement model class with type hints
3. Create `tests/models/test_new_model.py`
4. Update `pyproject.toml` if new dependencies needed
5. Document in [docs/MODELS.md](docs/MODELS.md)

### Adding a New Data Format

1. Create `src/cosmic_object_scanner/data/split_newformat.py`
2. Implement conversion utilities
3. Create `tests/data/test_split_newformat.py`
4. Document in [docs/DATASET.md](docs/DATASET.md)

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'cosmic_object_scanner'`

**Solution**:
```bash
# Ensure src is in PYTHONPATH or use uv
uv run python script.py

# Or activate venv
source .venv/bin/activate
cd /path/to/project  # Project root
```

### CUDA Issues

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use CPU: `device = torch.device('cpu')`

### Import Ordering

**Problem**: Pre-commit rejects commit due to import order

**Solution**:
```bash
uv run isort src/ tests/  # Fix imports
uv run black src/ tests/   # Format
git add .
git commit -m "..."
```

See more in [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md).
