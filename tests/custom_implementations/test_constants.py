"""Unit tests for custom_implementations.constants module."""

import pytest

from cosmic_object_scanner.custom_implementations.constants import (
    ANCHORS,
    batch_size,
    checkpoint_file,
    class_labels,
    device,
    epochs,
    image_size,
    learning_rate,
    load_model,
    s,
    save_model,
)


@pytest.mark.unit
class TestConstants:
    """Verify all constants have valid types and values."""

    def test_device_is_valid(self) -> None:
        assert device in ("cuda", "mps", "cpu")

    def test_anchors_structure(self) -> None:
        assert len(ANCHORS) == 3
        for scale in ANCHORS:
            assert len(scale) == 3
            for anchor in scale:
                assert len(anchor) == 2
                assert all(isinstance(v, float) for v in anchor)

    def test_image_size_positive(self) -> None:
        assert image_size > 0

    def test_grid_sizes(self) -> None:
        assert len(s) == 3
        assert s == [image_size // 32, image_size // 16, image_size // 8]

    def test_class_labels(self) -> None:
        assert class_labels == ["galaxy", "nebula", "star_cluster"]
        assert len(class_labels) == 3

    def test_hyperparameters(self) -> None:
        assert learning_rate > 0
        assert batch_size > 0
        assert epochs > 0

    def test_model_flags(self) -> None:
        assert isinstance(load_model, bool)
        assert isinstance(save_model, bool)

    def test_checkpoint_file(self) -> None:
        assert checkpoint_file.endswith(".pth.tar")
