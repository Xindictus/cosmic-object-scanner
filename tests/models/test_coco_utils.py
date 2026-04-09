"""Unit tests for models.coco_utils module."""

from typing import Any

import pytest
import torch

from cosmic_object_scanner.models.coco_utils import (
    ConvertCocoPolysToMask,
    convert_coco_poly_to_mask,
)


@pytest.mark.unit
class TestConvertCocoPolyToMask:
    """Tests for the convert_coco_poly_to_mask function."""

    def test_empty_segmentations(self) -> None:
        masks = convert_coco_poly_to_mask([], height=100, width=100)
        assert masks.shape == (0, 100, 100)
        assert masks.dtype == torch.uint8

    def test_output_type(self) -> None:
        # Simple polygon (triangle) inside a 100x100 image
        segmentations = [[[10.0, 10.0, 50.0, 10.0, 30.0, 50.0]]]
        masks = convert_coco_poly_to_mask(segmentations, height=100, width=100)
        assert isinstance(masks, torch.Tensor)
        assert masks.dtype == torch.uint8
        assert masks.shape[0] == 1
        assert masks.shape[1] == 100
        assert masks.shape[2] == 100


@pytest.mark.unit
class TestConvertCocoPolysToMaskCallable:
    """Tests for the ConvertCocoPolysToMask callable class."""

    def _make_image_and_target(self) -> tuple[Any, dict[str, Any]]:
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        target: dict[str, Any] = {
            "image_id": 1,
            "annotations": [
                {
                    "bbox": [10, 10, 30, 30],
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]],
                    "area": 900,
                    "iscrowd": 0,
                }
            ],
        }
        return img, target

    def test_returns_image_and_target(self) -> None:
        converter = ConvertCocoPolysToMask()
        img, target = self._make_image_and_target()
        out_img, out_target = converter(img, target)
        assert out_img is img
        assert "boxes" in out_target
        assert "labels" in out_target
        assert "masks" in out_target
        assert "area" in out_target
        assert "iscrowd" in out_target

    def test_boxes_tensor_shape(self) -> None:
        converter = ConvertCocoPolysToMask()
        img, target = self._make_image_and_target()
        _, out_target = converter(img, target)
        assert out_target["boxes"].shape == (1, 4)
        assert out_target["labels"].shape == (1,)

    def test_filters_crowd_annotations(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        target: dict[str, Any] = {
            "image_id": 1,
            "annotations": [
                {
                    "bbox": [10, 10, 30, 30],
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]],
                    "area": 900,
                    "iscrowd": 1,
                }
            ],
        }
        converter = ConvertCocoPolysToMask()
        _, out_target = converter(img, target)
        assert out_target["boxes"].shape[0] == 0
