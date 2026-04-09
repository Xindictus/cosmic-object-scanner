"""Unit tests for models.ultralytics_test module (pure functions)."""

from typing import Any

import pytest

from cosmic_object_scanner.models.ultralytics_test import (
    calculate_accuracy,
    calculate_iou,
    get_class,
    parse_yolo_label,
)


@pytest.mark.unit
class TestGetClass:
    """Tests for the get_class helper."""

    def test_galaxy(self) -> None:
        assert get_class(0) == "Galaxy"

    def test_nebula(self) -> None:
        assert get_class(1) == "Nebula"

    def test_star_cluster(self) -> None:
        assert get_class(2) == "Star Cluster"

    def test_invalid_class_raises(self) -> None:
        with pytest.raises(KeyError):
            get_class(99)


@pytest.mark.unit
class TestCalculateIoU:
    """Tests for the calculate_iou function."""

    def test_perfect_overlap(self) -> None:
        box = [0.0, 0.0, 10.0, 10.0]
        result = calculate_iou(box, box)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self) -> None:
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        result = calculate_iou(box1, box2)
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 5.0, 15.0, 15.0]
        result = calculate_iou(box1, box2)
        assert 0.0 < result < 1.0


@pytest.mark.unit
class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        precision, recall, f1 = calculate_accuracy(10, 0, 0)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_predictions(self) -> None:
        precision, recall, f1 = calculate_accuracy(0, 0, 5)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_mixed_results(self) -> None:
        # TP=5, FP=5, FN=5
        precision, recall, f1 = calculate_accuracy(5, 5, 5)
        assert precision == pytest.approx(0.5)
        assert recall == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


@pytest.mark.unit
class TestParseYoloLabel:
    """Tests for parse_yolo_label function."""

    def test_parses_single_annotation(self, tmp_path: Any) -> None:
        label_file = tmp_path / "label.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\n")

        result = parse_yolo_label(str(label_file), img_width=100, img_height=100)
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        # x_center=50, y_center=50, w=20, h=20 -> xmin=40, ymin=40, xmax=60, ymax=60
        assert result[0]["bbox"] == pytest.approx([40.0, 40.0, 60.0, 60.0])

    def test_parses_multiple_annotations(self, tmp_path: Any) -> None:
        label_file = tmp_path / "label.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        result = parse_yolo_label(str(label_file), img_width=200, img_height=200)
        assert len(result) == 2
        assert result[0]["class_id"] == 0
        assert result[1]["class_id"] == 1
