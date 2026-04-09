"""Unit tests for model utility functions."""

from typing import Any

import numpy as np
import pytest

from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import box_accuracy, calculate_iou


@pytest.mark.unit
class TestIoUCalculation:
    """Test suite for Intersection over Union (IoU) calculation."""

    def test_perfect_overlap(self) -> None:
        """Test IoU for perfectly overlapping boxes."""
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([0.0, 0.0, 10.0, 10.0])
        iou = calculate_iou(box1, box2)
        assert iou == pytest.approx(1.0), f"Expected 1.0, got {iou}"

    def test_no_overlap(self) -> None:
        """Test IoU for non-overlapping boxes."""
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([15.0, 15.0, 25.0, 25.0])
        iou = calculate_iou(box1, box2)
        assert iou == 0.0, f"Expected 0.0, got {iou}"

    def test_partial_overlap(self) -> None:
        """Test IoU for partially overlapping boxes."""
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([5.0, 5.0, 15.0, 15.0])
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 = 0.142857...
        iou = calculate_iou(box1, box2)
        expected = 25.0 / 175.0
        assert iou == pytest.approx(expected), f"Expected {expected}, got {iou}"

    def test_containment(self) -> None:
        """Test IoU when one box contains another."""
        box1 = np.array([0.0, 0.0, 20.0, 20.0])
        box2 = np.array([5.0, 5.0, 15.0, 15.0])
        # Intersection: 10x10 = 100
        # Union: 400 + 100 - 100 = 400
        # IoU = 100/400 = 0.25
        iou = calculate_iou(box1, box2)
        expected = 100.0 / 400.0
        assert iou == pytest.approx(expected), f"Expected {expected}, got {iou}"

    def test_iou_range(self) -> None:
        """Test that IoU is always in [0, 1] range."""
        boxes = [
            (np.array([0.0, 0.0, 10.0, 10.0]), np.array([0.0, 0.0, 10.0, 10.0])),
            (np.array([0.0, 0.0, 10.0, 10.0]), np.array([5.0, 5.0, 15.0, 15.0])),
            (np.array([0.0, 0.0, 10.0, 10.0]), np.array([15.0, 15.0, 25.0, 25.0])),
        ]
        for box1, box2 in boxes:
            iou = calculate_iou(box1, box2)
            assert 0.0 <= iou <= 1.0, f"IoU {iou} out of range [0, 1]"


@pytest.mark.unit
class TestBoxAccuracy:
    """Test suite for bounding box accuracy calculation."""

    def test_perfect_predictions(self) -> None:
        """Test accuracy when predictions perfectly match targets."""
        preds = [np.array([0.0, 0.0, 10.0, 10.0])]
        targets = [np.array([0.0, 0.0, 10.0, 10.0])]
        accuracy = box_accuracy(preds, targets)
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_no_matching_predictions(self) -> None:
        """Test accuracy when no predictions match targets."""
        preds = [np.array([0.0, 0.0, 10.0, 10.0])]
        targets = [np.array([30.0, 30.0, 40.0, 40.0])]
        accuracy = box_accuracy(preds, targets)
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"

    def test_empty_targets(self) -> None:
        """Test accuracy with empty target list."""
        preds = [np.array([0.0, 0.0, 10.0, 10.0])]
        targets: list[Any] = []
        accuracy = box_accuracy(preds, targets)
        assert accuracy == 0.0, f"Expected 0.0 for empty targets, got {accuracy}"

    def test_multiple_targets_partial_match(self) -> None:
        """Test accuracy with multiple targets and partial matches."""
        preds = [np.array([0.0, 0.0, 10.0, 10.0]), np.array([30.0, 30.0, 40.0, 40.0])]
        targets = [np.array([0.0, 0.0, 10.0, 10.0]), np.array([30.0, 30.0, 40.0, 40.0])]
        accuracy = box_accuracy(preds, targets)
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_iou_threshold_effect(self) -> None:
        """Test that IoU threshold affects accuracy calculation."""
        preds = [np.array([0.0, 0.0, 10.0, 10.0])]
        targets = [np.array([5.0, 5.0, 15.0, 15.0])]

        # With low threshold, should match
        accuracy_low = box_accuracy(preds, targets, iou_threshold=0.1)
        assert accuracy_low == 1.0, f"Expected 1.0 with low threshold, got {accuracy_low}"

        # With high threshold, should not match
        accuracy_high = box_accuracy(preds, targets, iou_threshold=0.9)
        assert accuracy_high == 0.0, f"Expected 0.0 with high threshold, got {accuracy_high}"


@pytest.mark.unit
class TestVisualizationFunctions:
    """Test suite for visualization utility functions."""

    def test_get_category_names(self) -> None:
        """Test that category names are correctly retrieved."""
        from cosmic_object_scanner.models.visualization import get_category_names

        category_names = get_category_names()
        assert isinstance(category_names, dict), "Expected dict return type"
        assert 0 in category_names, "Expected class 0 in categories"
        assert 1 in category_names, "Expected class 1 in categories"
        assert 2 in category_names, "Expected class 2 in categories"
        assert category_names[0] == "Galaxy", "Expected Galaxy at index 0"
        assert category_names[1] == "Nebula", "Expected Nebula at index 1"
        assert category_names[2] == "Star Cluster", "Expected Star Cluster at index 2"

    def test_get_category_names_immutability(self) -> None:
        """Test that modifying returned dict doesn't affect subsequent calls."""
        from cosmic_object_scanner.models.visualization import get_category_names

        names1 = get_category_names()
        names1_original = names1.copy()

        # Modify the returned dict
        names1[3] = "Modified"

        # Get fresh copy and verify it's not modified
        names2 = get_category_names()
        assert 3 not in names2, "Category names were modified unexpectedly"
        assert names2 == names1_original, "Fresh call returned modified categories"
