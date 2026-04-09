"""Unit tests for models.fasterrcnn_resnet50_fpn module."""

import numpy as np
import pytest

from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import (
    collate_fn,
    match_predictions_to_ground_truths,
)


@pytest.mark.unit
class TestMatchPredictions:
    """Tests for match_predictions_to_ground_truths."""

    def test_perfect_match(self) -> None:
        pred_boxes = [np.array([0.0, 0.0, 10.0, 10.0])]
        true_boxes = [np.array([0.0, 0.0, 10.0, 10.0])]
        pred_labels = [1]
        true_labels = [1]
        pred_scores = np.array([0.9])

        matched_pred, matched_true, matched_scores = match_predictions_to_ground_truths(
            pred_boxes, true_boxes, pred_labels, true_labels, pred_scores
        )
        assert matched_pred == [1]
        assert matched_true == [1]
        assert matched_scores == [0.9]

    def test_no_match(self) -> None:
        pred_boxes = [np.array([50.0, 50.0, 60.0, 60.0])]
        true_boxes = [np.array([0.0, 0.0, 10.0, 10.0])]
        pred_labels = [1]
        true_labels = [1]
        pred_scores = np.array([0.9])

        matched_pred, matched_true, matched_scores = match_predictions_to_ground_truths(
            pred_boxes, true_boxes, pred_labels, true_labels, pred_scores
        )
        assert matched_pred == [0]  # no-object class
        assert matched_true == [1]
        assert matched_scores == [0.0]

    def test_multiple_predictions(self) -> None:
        pred_boxes = [
            np.array([0.0, 0.0, 10.0, 10.0]),
            np.array([20.0, 20.0, 30.0, 30.0]),
        ]
        true_boxes = [
            np.array([0.0, 0.0, 10.0, 10.0]),
            np.array([20.0, 20.0, 30.0, 30.0]),
        ]
        pred_labels = [1, 2]
        true_labels = [1, 2]
        pred_scores = np.array([0.8, 0.9])

        matched_pred, matched_true, matched_scores = match_predictions_to_ground_truths(
            pred_boxes, true_boxes, pred_labels, true_labels, pred_scores
        )
        assert len(matched_pred) == 2
        assert len(matched_true) == 2

    def test_empty_predictions(self) -> None:
        matched_pred, matched_true, matched_scores = match_predictions_to_ground_truths(
            pred_boxes=[],
            true_boxes=[np.array([0.0, 0.0, 10.0, 10.0])],
            pred_labels=[],
            true_labels=[1],
            pred_scores=np.array([]),
        )
        assert matched_pred == [0]
        assert matched_true == [1]


@pytest.mark.unit
class TestCollateFn:
    """Tests for the collate_fn helper."""

    def test_collate_basic(self) -> None:
        batch = [("img1", {"target": 1}), ("img2", {"target": 2})]
        images, targets = collate_fn(batch)
        assert images == ("img1", "img2")
        assert targets == ({"target": 1}, {"target": 2})

    def test_collate_empty(self) -> None:
        result = collate_fn([])
        assert result == ()
