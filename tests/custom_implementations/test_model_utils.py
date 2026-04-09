"""Unit tests for custom_implementations.model_utils module."""

import pytest
import torch

from cosmic_object_scanner.custom_implementations.model_utils import YOLOLoss


@pytest.mark.unit
class TestYOLOLoss:
    """Tests for the YOLOLoss module."""

    def test_initialization(self) -> None:
        loss_fn = YOLOLoss()
        assert loss_fn.mse is not None
        assert loss_fn.bce is not None
        assert loss_fn.cross_entropy is not None
        assert loss_fn.sigmoid is not None

    def test_forward_returns_tensor(self) -> None:
        loss_fn = YOLOLoss()
        batch_size = 2
        num_anchors = 3
        grid_size = 13
        num_classes = 3

        # pred: [batch, anchors, grid, grid, 5+classes]
        pred = torch.randn(batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        target = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 6)
        # Set some cells as having objects
        target[0, 0, 5, 5, 0] = 1
        target[0, 0, 5, 5, 1:5] = torch.tensor([0.5, 0.5, 0.3, 0.3])
        target[0, 0, 5, 5, 5] = 1  # class label

        anchors = torch.tensor([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        loss = loss_fn(pred, target, anchors)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_zero_target_gives_scalar_loss(self) -> None:
        loss_fn = YOLOLoss()
        pred = torch.randn(1, 3, 13, 13, 8)
        target = torch.zeros(1, 3, 13, 13, 6)
        anchors = torch.tensor([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        loss = loss_fn(pred, target, anchors)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
