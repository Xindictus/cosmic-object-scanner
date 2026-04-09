"""Unit tests for custom TensorFlow model implementations."""

import pytest
import tensorflow as tf

from cosmic_object_scanner.custom_implementations.bbox_regressor import build_regressor
from cosmic_object_scanner.custom_implementations.classifier import (
    build_classifier,
    build_sequential_classifier,
)
from cosmic_object_scanner.custom_implementations.model import (
    build_classifier_head,
    build_feature_extractor,
    build_model,
    build_model_adaptor,
    build_regressor_head,
)


@pytest.mark.unit
class TestBuildSequentialClassifier:
    """Tests for the sequential classifier factory."""

    def test_returns_sequential_model(self) -> None:
        model = build_sequential_classifier()
        assert isinstance(model, tf.keras.models.Sequential)

    def test_custom_num_classes(self) -> None:
        model = build_sequential_classifier(num_classes=5, input_size=64)
        assert isinstance(model, tf.keras.models.Sequential)
        # Last layer should have 5 units
        assert model.layers[-1].units == 5

    def test_default_has_3_output_classes(self) -> None:
        model = build_sequential_classifier()
        assert model.layers[-1].units == 3


@pytest.mark.unit
class TestBuildClassifier:
    """Tests for the functional classifier builder."""

    def test_returns_tensor(self) -> None:
        inputs = tf.keras.Input(shape=(64, 64, 3))
        output = build_classifier(inputs, input_size_param=64)
        assert output is not None
        assert output.shape[-1] == 3  # CLASSES


@pytest.mark.unit
class TestBuildRegressor:
    """Tests for the bounding box regressor."""

    def test_returns_tensor(self) -> None:
        inputs = tf.keras.Input(shape=(64, 64, 1))
        output = build_regressor(inputs, input_size=64)
        assert output is not None
        assert output.shape[-1] == 4  # 4 bbox coordinates


@pytest.mark.unit
class TestBuildModel:
    """Tests for the combined model builder."""

    def test_feature_extractor_passthrough(self) -> None:
        inputs = tf.keras.Input(shape=(10,))
        result = build_feature_extractor(inputs)
        assert result is inputs

    def test_model_adaptor_flattens(self) -> None:
        inputs = tf.keras.Input(shape=(10,))
        result = build_model_adaptor(inputs)
        assert result is not None

    def test_classifier_head_output(self) -> None:
        inputs = tf.keras.Input(shape=(64,))
        result = build_classifier_head(inputs)
        assert result.shape[-1] == 3

    def test_regressor_head_output(self) -> None:
        inputs = tf.keras.Input(shape=(64,))
        result = build_regressor_head(inputs)
        assert result.shape[-1] == 4

    def test_full_model_build(self) -> None:
        inputs = tf.keras.Input(shape=(10,))
        model = build_model(inputs)
        assert isinstance(model, tf.keras.Model)
        assert len(model.outputs) == 2  # classification + regression heads
