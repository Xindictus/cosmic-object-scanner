"""Unit tests for custom_implementations.data_utils module."""

import pytest

from cosmic_object_scanner.custom_implementations.data_utils import (
    normalize_coordinates,
    test_transform,
    train_transform,
)


@pytest.mark.unit
class TestNormalizeCoordinates:
    """Tests for the normalize_coordinates function."""

    def test_already_normalized(self) -> None:
        coords = [0.5, 0.5, 0.3, 0.3]
        result = normalize_coordinates(coords)
        assert result == [0.5, 0.5, 0.3, 0.3]

    def test_clamps_values_above_one(self) -> None:
        coords = [1.5, 0.5, 2.0, 0.3]
        result = normalize_coordinates(coords)
        assert result[0] == 1.0
        assert result[2] == 1.0
        assert result[1] == 0.5
        assert result[3] == 0.3

    def test_preserves_extra_elements(self) -> None:
        coords = [0.5, 0.5, 0.3, 0.3, 2.0]
        result = normalize_coordinates(coords)
        assert len(result) == 5
        assert result[4] == 2.0  # extra element not clamped


@pytest.mark.unit
class TestTransforms:
    """Verify augmentation transforms are configured."""

    def test_train_transform_exists(self) -> None:
        assert train_transform is not None

    def test_test_transform_exists(self) -> None:
        assert test_transform is not None
