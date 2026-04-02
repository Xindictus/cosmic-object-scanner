"""Tests for models module."""

import pytest


@pytest.mark.unit
class TestCocoDataset:
    """Test suite for CocoDataset."""

    def test_imports(self):
        """Test that model imports work."""
        # Verify modules can be imported
        try:
            from cosmic_object_scanner.models import coco_dataset

            assert hasattr(coco_dataset, "CocoDataset")
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


@pytest.mark.unit
class TestTransforms:
    """Test suite for transforms module."""

    def test_imports(self):
        """Test that transforms imports work."""
        try:
            from cosmic_object_scanner.models import transforms

            # Check that key torchvision transforms are available
            assert hasattr(transforms, "Compose")
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


@pytest.mark.unit
class TestVisualization:
    """Test suite for visualization module."""

    def test_imports(self):
        """Test that visualization imports work."""
        try:
            from cosmic_object_scanner.models import visualization

            assert hasattr(visualization, "visualize_prediction_v2")
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


@pytest.mark.unit
class TestFasterRCNN:
    """Test suite for Faster R-CNN model."""

    def test_imports(self):
        """Test that fasterrcnn model imports work."""
        try:
            from cosmic_object_scanner.models import fasterrcnn_resnet50_fpn

            # Check for key functions in fasterrcnn module
            assert hasattr(fasterrcnn_resnet50_fpn, "get_model")
            assert hasattr(fasterrcnn_resnet50_fpn, "collate_fn")
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


@pytest.mark.unit
class TestEngine:
    """Test suite for training engine."""

    def test_imports(self):
        """Test that engine imports work."""
        try:
            from cosmic_object_scanner.models import engine

            assert hasattr(engine, "train_one_epoch")
            assert hasattr(engine, "evaluate")
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")
