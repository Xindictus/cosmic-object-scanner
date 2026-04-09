# Cosmic Object Scanner

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)

A deep learning project for automated object detection of stellar objects (galaxies, nebulae, and star clusters) in astronomical images. Implements multiple state-of-the-art detection architectures including Faster R-CNN and YOLOv3, trained on 5,000+ manually annotated images using Label Studio.

## 📋 Quick Navigation

- **[Full Report](Report.pdf)** — Technical details and comprehensive analysis
- **[Presentation](Presentation.pptx)** — Visual overview
- **[Architecture Guide](docs/ARCHITECTURE.md)** — Code structure and design decisions
- **[Model Details](docs/MODELS.md)** — Architecture specs and training procedures
- **[Evaluation Results](docs/EVALUATION.md)** — Performance metrics and benchmarks

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation & Running Tests

```bash
# Clone and setup
git clone <repo-url>
cd cosmic-object-scanner

# Install with uv (fast Python package manager)
uv sync --extra dev

# Run tests to verify installation
uv run pytest tests/ -v

# Run code quality checks
uv run mypy src/ --strict
uv run ruff check src/
uv run black --check src/
```

## ✨ Key Features

- **Dual Framework Support**: PyTorch (Faster R-CNN) + TensorFlow (Hybrid Models)
- **Multiple Architectures**:
  - Faster R-CNN with ResNet50 backbone
  - YOLOv3 from scratch (TensorFlow)
  - Ultralytics YOLOv8 (PyTorch)
  - TensorFlow hybrid model (Classifier + BBox Regressor)
- **COCO Format Support**: Full integration with COCOEval metrics
- **Production-Ready**: Type hints, strict mypy, pre-commit hooks, CI/CD pipeline
- **Advanced Data Augmentation**: Albumentations-based transformations

## 📊 Dataset Overview

- **5,000+ manually annotated images** via Label Studio
- **3 classes**: Galaxy, Nebula, Star Cluster
- **COCO JSON format** with bounding box annotations
- **Train/Val/Test split**: 70/15/15
- **Label Studio project** with full annotation metadata

See [docs/DATASET.md](docs/DATASET.md) for details.

## 🤖 Models

| Model | Framework | Type | Performance |
|-------|-----------|------|-------------|
| **Faster R-CNN ResNet50** | PyTorch | Two-stage | AP: 62.4%, AP@0.5: 85.1% |
| **YOLOv3 (Custom)** | TensorFlow | Single-stage | mAP: 58.7%, Precision: 89.2% |
| **Ultralytics YOLOv8** | PyTorch | Single-stage | Latest architecture |
| **Hybrid Classifier** | TensorFlow | Custom | Multi-class classifier |

See [docs/MODELS.md](docs/MODELS.md) for detailed descriptions.

## 💻 Usage

### Training Faster R-CNN

```bash
uv run python src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py
```

### Running YOLO (Ultralytics)

```bash
uv run python src/cosmic_object_scanner/models/ultralytics_yolo.py
```

### Data Processing

```bash
# Explore annotations
uv run python src/cosmic_object_scanner/data/annot_explore.py

# Convert/split datasets
from cosmic_object_scanner.data import split_coco, split_yolo
```

### Development

```bash
# Full quality check
uv run mypy src/ --strict              # Type checking
uv run ruff check src/ tests/          # Linting
uv run black --check src/ tests/       # Formatting
uv run pytest tests/ --cov             # Tests with coverage

# Setup pre-commit hooks
uv run pre-commit install

# Run before commit
uv run pre-commit run --all-files
```

## 📦 Installation Details

### With UV (Recommended)

UV is a fast Python package manager with lock file reproducibility:

```bash
# Install all dependencies (including dev tools)
uv sync --extra dev

# Run any command with automatic environment
uv run python script.py

# Or activate virtual environment
source .venv/bin/activate
```

### With Pip

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
cosmic-object-scanner/
├── src/cosmic_object_scanner/
│   ├── models/                    # PyTorch models & training
│   ├── custom_implementations/    # TensorFlow & scratch implementations
│   └── data/                      # Dataset utilities
├── tests/                         # Unit & integration tests
│   ├── custom_implementations/    # TF & scratch model tests
│   ├── models/                    # PyTorch model tests
│   └── data/                      # Data utility tests
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Code structure
│   ├── MODELS.md                  # Model details
│   ├── DATASET.md                 # Dataset information
│   ├── EVALUATION.md              # Results & metrics
│   └── DEV_GUIDE.md               # Development guide
├── experimental/                  # Experimental notebooks & data
├── pyproject.toml                 # UV + tool configuration
└── .pre-commit-config.yaml        # Git hooks
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete details.

## 📈 Evaluation Results

### Faster R-CNN (10 epochs)
- **AP**: 62.4%
- **AP@0.5**: 85.1%
- **AP@0.75**: 68.3%

### YOLOv3  (20 epochs)
- **mAP**: 58.7%
- **Precision**: 89.2%
- **Recall**: 71.3%

See [docs/EVALUATION.md](docs/EVALUATION.md) for per-class metrics and detailed analysis.

## 🧪 Testing

```bash
# Run all tests with coverage report
uv run pytest tests/ --cov=src/cosmic_object_scanner --cov-report=html

# View coverage in browser
open htmlcov/index.html      # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html     # Windows

# Run specific test file
uv run pytest tests/models/test_coco_dataset.py -v

# Run only fast tests (skip slow/integration)
uv run pytest tests/ -m "not slow" -v
```

## 🛠️ Development

### Code Quality Standards

- **Type Hints**: All public functions require type hints (mypy strict mode)
- **Linting**: Ruff enforces PEP8 and best practices
- **Formatting**: Black for consistent style (line length: 100)
- **Testing**: pytest with coverage tracking (target: 80%)
- **Pre-commit**: Blocks commits with violations

### Common Tasks

```bash
#  Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/ --fix

# Type check (strict)
uv run mypy src/ --strict

# Full validation before commit
uv run pre-commit run --all-files
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Pull request process
- Code style and conventions
- Adding new models or datasets
- Troubleshooting common issues

## 📚 Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Module organization, design decisions
- **[docs/MODELS.md](docs/MODELS.md)** — Model architectures, training procedures
- **[docs/DATASET.md](docs/DATASET.md)** — Dataset description, annotation format
- **[docs/EVALUATION.md](docs/EVALUATION.md)** — Metrics, benchmarks, per-class analysis
- **[docs/DEV_GUIDE.md](docs/DEV_GUIDE.md)** — Development setup, common tasks
- **[Report.pdf](Report.pdf)** — Complete technical report

## 📄 Requirements

See [pyproject.toml](pyproject.toml):

**Core Dependencies**:
- torch ≥2.0.0, torchvision ≥0.17.0 (PyTorch)
- tensorflow ≥2.13.0 (TensorFlow)
- opencv-python ≥4.8.0, albumentations ≥1.3.0 (Computer Vision)
- ultralytics ≥8.0.0 (YOLO)
- numpy, pandas, scikit-learn, matplotlib, seaborn

**Development**:
- ruff, black, isort (Linting & formatting)
- mypy (Type checking)
- pytest, pytest-cov (Testing)
- pre-commit (Git hooks)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🔗 References

- [COCO Dataset API](https://github.com/cocodataset/cocoapi)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [YOLOv3](https://arxiv.org/abs/1804.02767)
- [Label Studio](https://labelstud.io/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)

## 🤝 Support & Issues

1. Check documentation in `docs/`
2. Review [Report.pdf](Report.pdf) for technical details
3. Open an issue or submit a PR

---

**Python**: 3.12+ | **Status**: Production-Ready | **Last Updated**: April 2, 2026
