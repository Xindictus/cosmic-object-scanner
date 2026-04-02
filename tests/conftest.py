"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Return path to test data directory."""
    test_dir = Path(__file__).parent
    data_path = test_dir / "data"
    data_path.mkdir(exist_ok=True)
    return data_path


@pytest.fixture
def sample_image_path(data_dir: Path) -> Path:
    """Return path to sample test image (or create dummy)."""
    # For now, create a simple dummy path
    # In real testing, you'd have actual test images
    return data_dir / "sample_image.jpg"


@pytest.fixture
def sample_coco_annotation() -> dict:
    """Return sample COCO format annotation."""
    return {
        "info": {
            "description": "Test Dataset",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [{"id": 1, "name": "MIT"}],
        "images": [
            {
                "id": 1,
                "file_name": "test_image_001.jpg",
                "height": 512,
                "width": 512,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "Galaxy"},
            {"id": 2, "name": "Nebula"},
            {"id": 3, "name": "Star Cluster"},
        ],
    }


@pytest.fixture
def sample_yolo_annotation() -> dict:
    """Return sample YOLO format annotation."""
    return {
        "yaml_config": {
            "path": "/path/to/data",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 3,
            "names": ["Galaxy", "Nebula", "Star Cluster"],
        },
        "annotations": ["0 0.5 0.5 0.3 0.3", "1 0.2 0.2 0.1 0.15"],
    }


@pytest.fixture
def dummy_image_tensor():
    """Return dummy image tensor for model testing."""
    try:
        import torch

        return torch.randn(1, 3, 512, 512)
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def dummy_images_batch():
    """Return batch of dummy image tensors."""
    try:
        import torch

        return [torch.randn(3, 512, 512) for _ in range(2)]
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def dummy_targets():
    """Return dummy targets for model testing."""
    try:
        import torch

        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),
                "image_id": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 150.0, 150.0]]),
                "labels": torch.tensor([2]),
                "image_id": torch.tensor([1]),
            },
        ]
    except ImportError:
        pytest.skip("PyTorch not available")


def pytest_collection_modifyitems(config, items):
    """Mark all tests by default, allow filtering by marker."""
    for item in items:
        # Add markers for test categorization
        if "models" in str(item.fspath) or "data" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests if they take a long time
        # (Can be customized per test with @pytest.mark.slow)


# Register custom markers
def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "models: marks tests for models")
    config.addinivalue_line("markers", "data: marks tests for data utilities")
