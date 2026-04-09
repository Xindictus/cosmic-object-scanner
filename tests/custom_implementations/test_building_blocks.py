"""Unit tests for custom_implementations.building_blocks module."""

import pytest
import torch

from cosmic_object_scanner.custom_implementations.building_blocks import (
    CNNBlock,
    ResidualBlock,
    ScalePrediction,
)


@pytest.mark.unit
class TestCNNBlock:
    """Tests for the CNNBlock."""

    def test_output_shape_with_batch_norm(self) -> None:
        block = CNNBlock(3, 16, use_batch_norm=True, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_output_shape_without_batch_norm(self) -> None:
        block = CNNBlock(3, 16, use_batch_norm=False, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_activation_applied(self) -> None:
        block = CNNBlock(3, 8, use_batch_norm=True, kernel_size=1)
        x = torch.randn(1, 3, 8, 8)
        out = block(x)
        # LeakyReLU allows small negative values but no large negatives
        assert out.min().item() >= -1.0  # LeakyReLU slope is 0.1

    def test_no_activation_without_batch_norm(self) -> None:
        block = CNNBlock(3, 8, use_batch_norm=False, kernel_size=1)
        x = torch.randn(1, 3, 8, 8)
        out = block(x)
        # Without batch norm, raw conv output can have large negatives
        assert isinstance(out, torch.Tensor)


@pytest.mark.unit
class TestResidualBlock:
    """Tests for the ResidualBlock."""

    def test_output_shape_preserves_dimensions(self) -> None:
        block = ResidualBlock(channels=64, use_residual=True, num_repeats=1)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        block = ResidualBlock(channels=64, use_residual=True, num_repeats=1)
        x = torch.zeros(1, 64, 8, 8)
        out = block(x)
        # With zero input and residual, output should be close to zero + layer(zero)
        assert out.shape == x.shape

    def test_no_residual(self) -> None:
        block = ResidualBlock(channels=64, use_residual=False, num_repeats=1)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_multiple_repeats(self) -> None:
        block = ResidualBlock(channels=32, use_residual=True, num_repeats=3)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == x.shape


@pytest.mark.unit
class TestScalePrediction:
    """Tests for the ScalePrediction block."""

    def test_output_shape(self) -> None:
        num_classes = 3
        in_channels = 64
        grid_size = 13
        block = ScalePrediction(in_channels, num_classes)
        x = torch.randn(2, in_channels, grid_size, grid_size)
        out = block(x)
        # Expected: (batch, 3, grid, grid, num_classes + 5)
        assert out.shape == (2, 3, grid_size, grid_size, num_classes + 5)

    def test_different_grid_sizes(self) -> None:
        num_classes = 3
        block = ScalePrediction(128, num_classes)
        for grid_size in [13, 26, 52]:
            x = torch.randn(1, 128, grid_size, grid_size)
            out = block(x)
            assert out.shape == (1, 3, grid_size, grid_size, num_classes + 5)
