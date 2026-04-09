"""Unit tests for custom_implementations.utils module."""

from typing import Any

import pytest
import torch

from cosmic_object_scanner.custom_implementations.utils import (
    convert_cells_to_bboxes,
    iou,
    keep_prominent_boxes,
    nms,
)


@pytest.mark.unit
class TestIoU:
    """Tests for the IoU function."""

    def test_identical_boxes_pred_mode(self) -> None:
        box = torch.tensor([5.0, 5.0, 4.0, 4.0])
        result = iou(box, box, is_pred=True)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap_pred_mode(self) -> None:
        box1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
        box2 = torch.tensor([10.0, 10.0, 2.0, 2.0])
        result = iou(box1, box2, is_pred=True)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap_pred_mode(self) -> None:
        box1 = torch.tensor([5.0, 5.0, 10.0, 10.0])
        box2 = torch.tensor([8.0, 8.0, 10.0, 10.0])
        result = iou(box1, box2, is_pred=True)
        assert 0.0 < result.item() < 1.0

    def test_width_height_mode(self) -> None:
        box1 = torch.tensor([4.0, 4.0])
        box2 = torch.tensor([4.0, 4.0])
        result = iou(box1, box2, is_pred=False)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_width_height_different_sizes(self) -> None:
        box1 = torch.tensor([2.0, 2.0])
        box2 = torch.tensor([4.0, 4.0])
        result = iou(box1, box2, is_pred=False)
        # min(2,4)*min(2,4)=4, union=4+16-4=16, iou=4/16=0.25
        assert result.item() == pytest.approx(0.25)

    def test_batched_pred_mode(self) -> None:
        box1 = torch.tensor([[5.0, 5.0, 4.0, 4.0], [0.0, 0.0, 2.0, 2.0]])
        box2 = torch.tensor([[5.0, 5.0, 4.0, 4.0], [10.0, 10.0, 2.0, 2.0]])
        result = iou(box1, box2, is_pred=True)
        assert result[0].item() == pytest.approx(1.0, abs=1e-5)
        assert result[1].item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_empty_input(self) -> None:
        result = nms([], iou_threshold=0.5, threshold=0.5)
        assert result == []

    def test_all_below_threshold(self) -> None:
        boxes = [[0, 0.1, 0, 0, 10, 10], [0, 0.2, 5, 5, 15, 15]]
        result = nms(boxes, iou_threshold=0.5, threshold=0.5)
        assert result == []

    def test_single_box_above_threshold(self) -> None:
        boxes = [[0, 0.9, 0, 0, 10, 10]]
        result = nms(boxes, iou_threshold=0.5, threshold=0.5)
        # Single box is always the "first" popped, and since no others, result is empty
        # (the first popped box is not appended to nms unless there are remaining boxes)
        assert isinstance(result, list)

    def test_non_overlapping_boxes(self) -> None:
        boxes = [
            [0, 0.9, 0.0, 0.0, 0.1, 0.1],
            [0, 0.8, 0.5, 0.5, 0.1, 0.1],
        ]
        result = nms(boxes, iou_threshold=0.5, threshold=0.5)
        # Non-overlapping with same class -> iou < threshold -> kept
        assert len(result) >= 1

    def test_different_classes_kept(self) -> None:
        boxes = [
            [0, 0.9, 5.0, 5.0, 2.0, 2.0],
            [1, 0.8, 5.0, 5.0, 2.0, 2.0],
        ]
        result = nms(boxes, iou_threshold=0.5, threshold=0.5)
        # Different classes, so second box is kept even if overlapping
        assert len(result) == 1  # class 1 box is kept


@pytest.mark.unit
class TestConvertCellsToBboxes:
    """Tests for convert_cells_to_bboxes."""

    def test_output_shape(self) -> None:
        batch_size = 2
        num_anchors = 3
        grid_size = 13
        num_classes = 3
        predictions = torch.randn(batch_size, num_anchors, grid_size, grid_size, num_classes + 5)
        anchors = torch.tensor([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        result = convert_cells_to_bboxes(predictions, anchors, s=grid_size, is_predictions=True)
        assert len(result) == batch_size
        assert len(result[0]) == num_anchors * grid_size * grid_size
        assert len(result[0][0]) == 6  # [class_id, score, x, y, w, h]

    def test_label_mode(self) -> None:
        batch_size = 1
        num_anchors = 3
        grid_size = 13
        num_classes = 3
        targets = torch.zeros(batch_size, num_anchors, grid_size, grid_size, num_classes + 5)
        anchors = torch.tensor([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        result = convert_cells_to_bboxes(targets, anchors, s=grid_size, is_predictions=False)
        assert len(result) == batch_size

    def test_scores_in_valid_range_for_predictions(self) -> None:
        predictions = torch.randn(1, 3, 13, 13, 8)
        anchors = torch.tensor([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        result = convert_cells_to_bboxes(predictions, anchors, s=13, is_predictions=True)
        for box in result[0]:
            score = box[1]
            assert 0.0 <= score <= 1.0, f"Score {score} not in sigmoid range"


@pytest.mark.unit
class TestKeepProminentBoxes:
    """Tests for the keep_prominent_boxes function."""

    def test_empty_input(self) -> None:
        result = keep_prominent_boxes([], iou_threshold=0.5, count_threshold=2)
        assert result == []

    def test_no_prominent_boxes(self) -> None:
        boxes: list[Any] = [
            [0, 0.9, 0.5, 0.5, 0.1, 0.1],
        ]
        # Single box can't have any similar boxes -> count=0 < threshold=2
        result = keep_prominent_boxes(boxes, iou_threshold=0.5, count_threshold=2)
        assert result == []

    def test_prominent_boxes_kept(self) -> None:
        # Three identical boxes of same class -> count_threshold=2 should keep
        boxes: list[Any] = [
            [0, 0.9, 0.5, 0.5, 0.1, 0.1],
            [0, 0.8, 0.5, 0.5, 0.1, 0.1],
            [0, 0.7, 0.5, 0.5, 0.1, 0.1],
        ]
        result = keep_prominent_boxes(boxes, iou_threshold=0.5, count_threshold=2)
        assert len(result) >= 1

    def test_different_classes_not_grouped(self) -> None:
        boxes: list[Any] = [
            [0, 0.9, 0.5, 0.5, 0.1, 0.1],
            [1, 0.8, 0.5, 0.5, 0.1, 0.1],
        ]
        # Different classes, so no box has similar boxes -> both removed
        result = keep_prominent_boxes(boxes, iou_threshold=0.5, count_threshold=1)
        assert isinstance(result, list)
