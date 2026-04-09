"""Custom model implementations."""

from typing import Any

import tensorflow as tf

CLASSES = 3


def build_feature_extractor(inputs: Any) -> Any:
    return inputs


def build_model_adaptor(inputs: Any) -> Any:
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    return x


def build_classifier_head(inputs: Any) -> Any:
    return tf.keras.layers.Dense(CLASSES, activation="softmax", name="classifier_head")(inputs)


def build_regressor_head(inputs: Any) -> Any:
    return tf.keras.layers.Dense(units=4, name="regressor_head")(inputs)


def build_model(inputs: Any) -> Any:
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head])

    return model
