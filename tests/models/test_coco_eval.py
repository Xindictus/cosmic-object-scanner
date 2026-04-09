"""Unit tests for models.coco_eval module."""

import pytest
import torch

from cosmic_object_scanner.models.coco_eval import convert_to_xywh


@pytest.mark.unit
class TestConvertToXywh:
    """Tests for convert_to_xywh utility."""

    def test_basic_conversion(self) -> None:
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        result = convert_to_xywh(boxes)
        assert result[0].tolist() == pytest.approx([10.0, 10.0, 40.0, 40.0])

    def test_multiple_boxes(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 30.0]])
        result = convert_to_xywh(boxes)
        assert result.shape == (2, 4)
        assert result[0].tolist() == pytest.approx([0.0, 0.0, 10.0, 10.0])
        assert result[1].tolist() == pytest.approx([5.0, 5.0, 15.0, 25.0])

    def test_zero_size_box(self) -> None:
        boxes = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        result = convert_to_xywh(boxes)
        assert result[0].tolist() == pytest.approx([5.0, 5.0, 0.0, 0.0])
