# Contributing Guide

Thank you for your interest in contributing to Cosmic Object Scanner! This guide outlines the process for getting involved and creating a pull request.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We value diverse perspectives and welcome contributions from everyone.

## Ways to Contribute

1. **Bug Reports**: Found an issue? Open a GitHub issue with reproduction steps
2. **Feature Requests**: Have an idea? Suggest it in a GitHub discussion or issue
3. **Code**: Fix bugs or implement features with a PR
4. **Documentation**: Improve guides, API docs, or examples
5. **Testing**: Add tests or improve test coverage
6. **Performance**: Profile and optimize code

## Development Setup

See [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md) for detailed setup instructions.

Quick start:
```bash
git clone <repo-url>
cd cosmic-object-scanner
uv sync --dev
uv run pre-commit install
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/description-of-change
# Example: git checkout -b feature/add-model-ensembling
```

### 2. Make Your Changes

- Keep changes focused and atomic
- Write clear, descriptive commit messages
- Add tests for new code
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format and lint
uv run black src/ tests/
uv run isort src/ tests/
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/ --strict

# Tests with coverage
uv run pytest tests/ --cov=src/cosmic_object_scanner --cov-fail-under=60

# All checks (same as CI)
uv run pre-commit run --all-files
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: Add model ensembling support"
git push origin feature/add-model-ensembling
```

### 5. Open Pull Request

On GitHub:
- Title: Brief description (e.g., "Add model ensembling support")
- Description: Explain what and why
- Link related issues: "Closes #123"
- Request reviewers
- Ensure CI passes

### 6. Address Review Feedback

- Respond to comments
- Make requested changes
- Push new commits
- Re-request review

### 7. Merge

Once approved, we'll merge your PR. Thank you!

## Contribution Guidelines

### Code Style

- **Format**: Black (100 char line length)
- **Linting**: Ruff (PEP8 + best practices)
- **Imports**: isort (sorted alphabetically within groups)
- **Type Hints**: MyPy strict mode required

### Commits

Write clear, descriptive commit messages:

```
feat: Add feature description
- Details about what was added
- Why it was necessary
- Related issue: #123
```

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test additions
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance

### Testing

- Add tests for all new code
- Minimum 60% coverage (target 80%)
- Mark slow tests with `@pytest.mark.slow`

```python
def test_my_feature():
    """Test that my feature works."""
    result = my_feature()
    assert result is not None
```

### Documentation

- Add/update docstrings (Google style)
- Update relevant .md files in `docs/`
- Add examples for complex features

```python
def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int
) -> Dict[str, float]:
    """Train model for specified epochs.

    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        epochs: Number of training epochs

    Returns:
        Dictionary with training metrics

    Raises:
        ValueError: If epochs <= 0
    """
```

### Performance

- Profile before optimizing
- Include performance metrics in PR
- Discuss trade-offs

### Security

- Never commit secrets or credentials
- Use environment variables for config
- Validate user input
- Use `.gitignore` for sensitive files

## Running Tests Locally

Before pushing your changes, run the full test suite locally to catch issues early.

### Quick Test Run

```bash
# Run tests only
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/models/ -v

# Run specific test
uv run pytest tests/test_imports.py::test_can_import_all_modules -v

# Run with markers
uv run pytest tests/ -m "not slow" -v  # Skip slow tests
uv run pytest tests/ -m "models" -v    # Run only model tests
```

### Full Quality Check (Same as CI)

Run all checks to match what CI will test:

```bash
# 1. Format code
uv run black src/ tests/

# 2. Check linting
uv run ruff check src/ tests/ --fix

# 3. Check import ordering
uv run isort src/ tests/

# 4. Type checking
uv run mypy src/ --strict

# 5. Run tests
uv run pytest tests/ -v --cov=src/cosmic_object_scanner

# Or run all at once (same order as pre-commit)
uv run pre-commit run --all-files
```

### Using Pre-commit Locally

Set up pre-commit hooks to automatically check before commit:

```bash
# Already installed during setup, but can reinstall:
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files

# Run on staged files before commit
uv run pre-commit run  # Auto-runs on git commit if installed

# Uninstall if needed
uv run pre-commit uninstall
```

### Understanding Test Output

**Successful run**:
```
============================= test session starts ==============================
collected 12 items

tests/test_imports.py::test_can_import_all_modules PASSED       [ 50%]
tests/test_imports.py::test_imports_have_no_side_effects PASSED [100%]

============================== 12 passed in 0.23s ===============================
```

**Failed test**:
```
FAILED tests/test_imports.py::test_something - AssertionError: expected True
```

**Skipped test**:
```
SKIPPED tests/models/test_model_validation.py::test_fastercnn_model_initialization - PyTorch not available
```

### Common Issues & Troubleshooting

#### "ModuleNotFoundError: No module named 'cosmic_object_scanner'"

**Solution**: Run from project root with UV
```bash
cd /path/to/cosmic-object-scanner
uv run pytest tests/
```

#### "ImportError: cannot import name 'X' from 'cosmic_object_scanner'"

**Cause**: Direct import issue
**Solution**:
1. Check the import path is correct
2. Verify the module exists in `src/cosmic_object_scanner/`
3. Run: `uv run black . && uv run isort .`

#### "MyPy strict mode errors"

**Expected**: Some third-party library errors may appear
**Solution**: Filter with `--ignore-missing-imports` or add type stub packages

```bash
uv run mypy src/ --strict  # See all errors
uv run mypy src/ --ignore-missing-imports  # Suppress untyped lib errors
```

#### "Tests hang or timeout"

**Cause**: Usually slow tests with large datasets
**Solution**:
```bash
# Skip slow tests
uv run pytest tests/ -m "not slow"

# Set timeout
uv run pytest tests/ --timeout=10
```

#### "Coverage below minimum"

**View coverage**:
```bash
uv run pytest tests/ --cov=src/cosmic_object_scanner --cov-report=html
# Opens htmlcov/index.html in browser
```

**Improve coverage**: Add tests for uncovered lines

## GitHub Actions CI/CD

### What Runs on Every Push

Two workflows automatically run:

1. **lint-and-test.yml** (Every push/PR)
   - Ruff linting
   - Black formatting check
   - isort import ordering
   - MyPy type checking
   - pytest test suite

2. **model-validation.yml** (Nightly 1 AM UTC)
   - Verifies model loading
   - Testing on GPU if available
   - Longer-running validations

### View CI Results

On GitHub PRs:
1. Go to your PR
2. Scroll to "Checks" section
3. Click "lint-and-test" to see detailed logs
4. Fix any failures locally and push again

### Debug CI Failures Locally

If CI fails but tests pass locally:

```bash
# Check Python version (CI uses 3.12)
python --version

# Check installed versions match
uv pip list

# Run specific failing test
uv run pytest tests/test_imports.py::test_name -v

# Run with same settings as CI
export PYTHONPATH=/home/user/cosmic-object-scanner/src:$PYTHONPATH
uv run pytest tests/ -v
```

## Review Process

### What We Look For

✅ **Good PRs**:
- Clear, descriptive title and description
- Focused changes (one feature/fix per PR)
- Tests and documentation included
- Code quality checks pass
- Responsive to feedback

❌ **Issues That Cause Delays**:
- Unclear description
- Mixed unrelated changes
- No tests
- Type checking failures
- Outdated documentation

### Feedback

We provide constructive feedback to help improve code. Remember:
- Comments are about code, not people
- We value learning and growth
- Questions are encouragement to discuss

## Reporting Issues

When reporting bugs:

1. **Check existing issues** - Maybe it's already reported
2. **Provide details**:
   - OS and Python version
   - Exact error message
   - Steps to reproduce
   - Environment (GPU/CPU, dependencies)
   - Code snippet if applicable

3. **Example**:
   ```
   **Describe the bug**
   Model training crashes when using batch_size > 4

   **To Reproduce**
   1. Run: python train.py --batch-size 8
   2. Error occurs at epoch 2

   **Environment**
   - OS: Ubuntu 20.04
   - Python: 3.12.0
   - CUDA: 11.8
   - GPU: NVIDIA T4

   **Error Message**
   RuntimeError: CUDA out of memory
   ```

## Feature Requests

Suggest features by:
1. Opening a GitHub issue with "enhancement" label
2. Describing use case and motivation
3. Proposing implementation approach
4. Discussing trade-offs

## Documentation Improvements

Help us improve docs:
- Fix typos or unclear sections
- Add examples
- Clarify complex concepts
- Suggest better organization

## Getting Help

- **Questions**: Open a discussion or issue
- **Getting Started**: See [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md)
- **Architecture**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Models**: See [docs/MODELS.md](docs/MODELS.md)

## Community

- Respect diverse perspectives
- Help other contributors
- Share knowledge and ideas
- Have fun building together!

## License

By contributing, you agree that your contributions are licensed under the project's MIT License.

---

**Thank you for contributing to Cosmic Object Scanner!** 🌟
