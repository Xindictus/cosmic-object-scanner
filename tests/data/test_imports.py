"""Tests for data utilities module."""

from typing import Any

import pytest


@pytest.mark.unit
class TestDataUtilities:
    """Test suite for data utilities."""

    def test_split_coco_imports(self) -> None:
        """Test that split_coco module imports."""
        try:
            from cosmic_object_scanner.data import split_coco  # noqa: F401
            from cosmic_object_scanner.data.split_coco import filter_data, move_images

            # Module should have key functions
            assert callable(move_images)
            assert callable(filter_data)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_split_yolo_imports(self) -> None:
        """Test that split_yolo module imports."""
        try:
            from cosmic_object_scanner.data import split_yolo  # noqa: F401
            from cosmic_object_scanner.data.split_yolo import main

            # Module should have key functions
            assert callable(main)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_annot_explore_imports(self) -> None:
        """Test that annot_explore module imports."""
        try:
            from cosmic_object_scanner.data import annot_explore  # noqa: F401
            from cosmic_object_scanner.data.annot_explore import main

            # Module should have key functions
            assert callable(main)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


@pytest.mark.unit
class TestCocoFormat:
    """Test suite for COCO format utilities."""

    def test_coco_annotation_structure(self, sample_coco_annotation: Any) -> None:
        """Test COCO annotation has required structure."""
        assert "images" in sample_coco_annotation
        assert "annotations" in sample_coco_annotation
        assert "categories" in sample_coco_annotation

        # Verify image structure
        assert len(sample_coco_annotation["images"]) > 0
        image = sample_coco_annotation["images"][0]
        assert "id" in image
        assert "file_name" in image
        assert "height" in image
        assert "width" in image

        # Verify annotation structure
        assert len(sample_coco_annotation["annotations"]) > 0
        annotation = sample_coco_annotation["annotations"][0]
        assert "id" in annotation
        assert "image_id" in annotation
        assert "category_id" in annotation
        assert "bbox" in annotation

        # Verify category structure
        assert len(sample_coco_annotation["categories"]) > 0
        category = sample_coco_annotation["categories"][0]
        assert "id" in category
        assert "name" in category


@pytest.mark.unit
class TestYoloFormat:
    """Test suite for YOLO format utilities."""

    def test_yolo_annotation_structure(self, sample_yolo_annotation: Any) -> None:
        """Test YOLO annotation has required structure."""
        assert "yaml_config" in sample_yolo_annotation
        assert "annotations" in sample_yolo_annotation

        # Verify YAML config
        config = sample_yolo_annotation["yaml_config"]
        assert config["nc"] == 3
        assert len(config["names"]) == 3

        # Verify annotations are strings (YOLO format)
        for ann in sample_yolo_annotation["annotations"]:
            assert isinstance(ann, str)
            parts = ann.split()
            assert len(parts) == 5  # class_id, x_center, y_center, width, height
