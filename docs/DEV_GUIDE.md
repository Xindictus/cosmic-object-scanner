# Development Guide

## Getting Started

### Prerequisites

- Python 3.12+ (download from [python.org](https://www.python.org/downloads/))
- Git
- UV package manager (installed separately or via pip)
- (Optional) CUDA 11.8+ for GPU support

### Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd cosmic-object-scanner

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --dev

# Verify installation
uv run python --version
uv run pytest --version
```

## Development Workflow

### Creating a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/description

# Example: git checkout -b feature/add-inference-server
```

### Making Changes

```bash
# Edit code
vi src/cosmic_object_scanner/models/new_module.py

# Add tests
vi tests/models/test_new_module.py

# Run validation
uv run pytest tests/models/test_new_module.py -v
```

### Pre-Commit Validation

```bash
# Install pre-commit hooks (one-time)
uv run pre-commit install

# Run validation before committing
uv run pre-commit run --all-files

# Common issues and fixes:
# - isort auto-fixes import ordering
# - black auto-formats code
# - mypy reports type errors (fix manually)
# - ruff suggests fixes (--fix flag)
```

### Committing Changes

```bash
# Stage changes
git add src/ tests/ docs/

# Commit with descriptive message
git commit -m "feat: Add inference server for real-time predictions"

# If pre-commit fails:
uv run black src/
uv run isort src/
uv run mypy src/ --strict  # Fix issues manually
git add .
git commit -m "..."
```

### Pushing and Creating PR

```bash
# Push to remote
git push origin feature/description

# Create PR on GitHub
# - Link related issues
# - Add description and motivation
# - Request reviewers
```

## Code Quality Standards

### Type Hints

All public functions must have type hints. Mypy runs in strict mode.

**Good**:

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int
) -> Dict[str, float]:
    """Train the model."""
    ...
```

**Bad** (missing type hints):

```python
def train_model(model, train_loader, num_epochs):
    """Train the model."""
    ...
```

### Docstrings

Use Google-style docstrings for clarity:

```python
def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate Intersection over Union between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) with boxes [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) with boxes [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)

    Raises:
        ValueError: If boxes have incorrect shape

    Example:
        >>> boxes1 = torch.tensor([[0, 0, 10, 10]])
        >>> boxes2 = torch.tensor([[5, 5, 15, 15]])
        >>> iou = calculate_iou(boxes1, boxes2)
        >>> assert iou[0, 0] > 0.1
    """
    ...
```

### Linting & Formatting

```bash
# Check format (don't modify)
uv run black --check src/ tests/

# Auto-format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check src/ tests/ --fix

# Sort imports
uv run isort src/ tests/

# Type checking (requires fixes)
uv run mypy src/ --strict
```

### Testing

Write tests for all new code:

```python
# tests/models/test_new_module.py
import pytest
import torch
from cosmic_object_scanner.models.new_module import MyModel

class TestMyModel:
    """Test suite for MyModel."""

    @pytest.fixture
    def model(self):
        """Create a model instance."""
        return MyModel()

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        assert output.shape == (2, 3, 256, 256)

    def test_invalid_input(self, model):
        """Test model handles invalid input gracefully."""
        with pytest.raises(ValueError):
            model(torch.randn(2, 1, 256, 256))  # Wrong channels
```

**Coverage**: All public functions should have tests. Target coverage: 80%.

```bash
# Run tests with coverage report
uv run pytest tests/ --cov=src/cosmic_object_scanner --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Common Development Tasks

### Adding a New Model

1. **Create model file**:

   ```bash
   touch src/cosmic_object_scanner/models/my_model.py
   ```

2. **Implement model**:

   ```python
   from typing import Dict, Optional
   import torch
   from torch import nn

   class MyModel(nn.Module):
       """Description of my model."""

       def __init__(self, num_classes: int = 3) -> None:
           super().__init__()
           self.backbone = nn.Sequential(...)
           self.head = nn.Sequential(...)

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass."""
           features = self.backbone(x)
           output = self.head(features)
           return output
   ```

3. **Add to package**:

   ```python
   # src/cosmic_object_scanner/models/__init__.py
   from cosmic_object_scanner.models.my_model import MyModel
   ```

4. **Write tests**:

   ```bash
   touch tests/models/test_my_model.py
   ```

5. **Document**:
   - Update [docs/MODELS.md](docs/MODELS.md)
   - Add example usage in comments

6. **Validate**:

   ```bash
   uv run mypy src/ --strict
   uv run pytest tests/models/test_my_model.py -v --cov
   ```

### Adding a New Data Format

1. **Create utility**:

   ```bash
   touch src/cosmic_object_scanner/data/split_newformat.py
   ```

2. **Implement converter**:

   ```python
   def convert_to_coco(
       input_path: str,
       output_path: str,
       split_ratio: float = 0.7
   ) -> None:
       """Convert NewFormat dataset to COCO format."""
       ...
   ```

3. **Add tests**:

   ```python
   def test_convert_to_coco(tmp_path):
       """Test conversion produces valid COCO JSON."""
       output_file = tmp_path / "result.json"
       convert_to_coco("input", str(tmp_path), split_ratio=0.8)
       assert output_file.exists()
       # Validate COCO format
       ...
   ```

4. **Document**:
   - Update [docs/DATASET.md](docs/DATASET.md)

### Running Experiments

```bash
# Run training with specific config
uv run python src/cosmic_object_scanner/models/fasterrcnn_resnet50_fpn.py

# Log results
# Edit script to save metrics to CSV:
# results.csv, learning_curves/, checkpoints/
```

### Debugging

```bash
# Enable verbose logging
uv run python -v script.py

# Use debugger
import pdb; pdb.set_trace()  # Add breakpoint

# Or use pytest with pdb
uv run pytest tests/ -v --pdb
```

### Performance Profiling

```bash
# Profile memory usage
import tracemalloc
tracemalloc.start()
# ... run code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current/1e6}MB; Peak: {peak/1e6}MB")

# Profile execution time
import time
start = time.time()
# ... run code ...
end = time.time()
print(f"Execution time: {end-start:.2f}s")
```

## GitHub Actions CI/CD

### Workflows

**`.github/workflows/lint-and-test.yml`** (on every PR):

1. Lint with ruff
2. Format check with black
3. Type check with mypy
4. Run tests with coverage

**`.github/workflows/model-validation.yml`** (nightly):

1. Train Faster R-CNN for 1 epoch
2. Validate inference
3. Report performance

### Checking CI Status

- View results on PR page
- Re-run failed jobs with "Re-run failed jobs"
- Check logs for error details

### Local Testing of CI

```bash
# Simulate CI locally
uv run ruff check src/ tests/
uv run black --check src/ tests/
uv run mypy src/ --strict
uv run pytest tests/ --cov
```

### Complete Local CI/CD Pipeline

Run the full quality check pipeline exactly as GitHub Actions does:

```bash
#!/bin/bash
# Run full CI pipeline locally

echo "Step 1: Ruff linting..."
uv run ruff check src/ tests/ || exit 1

echo "Step 2: Black formatting check..."
uv run black --check src/ tests/ || exit 1

echo "Step 3: isort import check..."
uv run isort --check-only src/ tests/ || exit 1

echo "Step 4: MyPy type checking..."
uv run mypy src/ --strict || exit 1

echo "Step 5: Pytest tests..."
uv run pytest tests/ -v --cov=src/cosmic_object_scanner --cov-fail-under=60 || exit 1

echo "All checks passed! Ready to push."
```

Or run all in one command:

```bash
uv run pre-commit run --all-files && \
uv run pytest tests/ -v --cov=src/cosmic_object_scanner
```

### Fixing CI Failures

**Ruff violations**:

```bash
uv run ruff check src/ tests/ --fix
```

**Black formatting**:

```bash
uv run black src/ tests/
```

**isort import ordering**:

```bash
uv run isort src/ tests/
```

**MyPy type errors** (must fix manually):

```bash
uv run mypy src/ --strict
# Fix reported errors in code, then re-run
```

**Test failures**:

```bash
# Run failing test with verbose output
uv run pytest tests/path/to/test.py::test_name -v

# Run with pdb debugger
uv run pytest tests/path/to/test.py -v --pdb
```

### Pre-Commit Hooks Explained

Hooks run automatically before commit (if installed):

1. **Ruff** - Fast Python linter
   - Checks for common errors (F-series)
   - Checks code style (E/W-series)
   - Auto-fixes with `--fix` flag

2. **Black** - Code formatter
   - Enforces consistent style
   - 100-character line length
   - Non-configurable (by design)

3. **isort** - Import sorter
   - Sorts imports alphabetically
   - Groups: future, stdlib, third-party, local
   - Matches Black's line length

4. **MyPy** - Static type checker
   - Runs in strict mode
   - Requires type hints on all functions
   - Catches many common bugs

5. **End-of-file-fixer** - File cleaning
   - Ensures single newline at end
   - Removes trailing whitespace

### GitHub Workflows Reference

#### lint-and-test.yml (On Every Push/PR)

```yaml
name: Lint and Test

on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.12']
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v2

      - name: Lint with Ruff
        run: uv run ruff check src/ tests/

      - name: Check formatting
        run: uv run black --check src/ tests/

      - name: Type check
        run: uv run mypy src/ --strict

      - name: Run tests
        run: uv run pytest tests/ -v --cov
```

#### model-validation.yml (Nightly)

```yaml
name: Model Validation

on:
  schedule:
    - cron: '0 1 * * *'  # 1 AM UTC daily

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v2

      - name: Run model validation tests
        run: uv run pytest tests/models/test_model_validation.py -v
```

## Documentation

### Writing Documentation

- **Code comments**: Explain the "why", not the "what"
- **Docstrings**: Function/class documentation
- **README**: Project-level overview
- **docs/**: Detailed guides for specific topics

### Documentation Build

```bash
# Build docs (if using Sphinx)
cd docs
make html

# Or manually create markdown files in docs/
# - ARCHITECTURE.md
# - MODELS.md
# - DATASET.md
# - EVALUATION.md
```

## Useful Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run specific test
uv run pytest tests/models/test_coco_dataset.py::test_load_single_image -v

# Run tests with specific marker
uv run pytest tests/ -m "not slow" -v

# Run with coverage for specific file
uv run pytest tests/models/ --cov=src/cosmic_object_scanner/models

# View what would be committed
uv run pre-commit run --all-files --dry-run

# Update dependencies
uv sync --upgrade

# Check for security issues
pip-audit  # or use built-in uv check (future)

# Benchmark model
uv run python -m torch.utils.bottleneck src/cosmic_object_scanner/models/model.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'cosmic_object_scanner'"

**Solution**:

```bash
# Make sure you're in the project root
cd /path/to/cosmic-object-scanner

# Use uv run (includes proper path setup)
uv run python script.py

# Or activate venv and set PYTHONPATH
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:$PYTHONPATH"
python script.py
```

### Type checking fails with "error: Skipping analyzing ..."

```bash
# Install stubs if needed
uv add --dev types-PyYAML types-requests

# Or allow untyped imports for specific modules
# Add to pyproject.toml:
# [[tool.mypy.overrides]]
# module = "untyped_module"
# ignore_missing_imports = true
```

### Tests hang or timeout

```bash
# Run with timeout
uv run pytest tests/ --timeout=300

# Or run with verbose output to see progress
uv run pytest tests/ -v --tb=short
```

### Pre-commit hooks too slow

```bash
# Check which hook is slow
uv run pre-commit run --all-files --verbose

# Skip hooks for now (not recommended)
git commit --no-verify  # Use with caution
```

## Best Practices

### Code Organization
- One class/function per file if complex
- Group related utilities in a module
- Keep module files < 500 lines

### Testing
- Write tests as you code
- Use fixtures for setup/teardown
- Mock external dependencies
- Test both happy path and edge cases

### Performance
- Profile before optimizing
- Use vectorized operations (numpy, torch)
- Avoid unnecessary copies
- Cache expensive computations

### Security
- Never commit credentials/secrets
- Use `.gitignore` for sensitive files
- Validate user input
- Use environment variables for config

### Collaboration
- Write clear commit messages
- Link related issues in PRs
- Request specific reviewers
- Respond to review feedback promptly

## Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Git Workflow](https://www.atlassian.com/git/tutorials)
- [Python Type Hints](https://docs.python.org/3.12/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

## Support

For questions or issues:
1. Check relevant documentation in `docs/`
2. Search existing GitHub issues
3. Open a new issue with:
   - Clear description of problem
   - Steps to reproduce
   - Environment info (OS, Python version, etc.)
   - Relevant error messages
4. Submit PR with fix

---

**Happy coding!** 🚀
