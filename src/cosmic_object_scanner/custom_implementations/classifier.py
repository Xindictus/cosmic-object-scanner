from typing import Any

import tensorflow as tf

CLASSES = 3
input_size = 608  # Default image size


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", input_shape=(608, 608, 3)),
        tf.keras.layers.AveragePooling2D(2, 2),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.AveragePooling2D(2, 2),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
        tf.keras.layers.AveragePooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(CLASSES, activation="softmax"),
    ]
)

CLASSES = 3


def build_classifier(inputs: tf.keras.Input, input_size_param: int = 608) -> Any:
    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, activation="relu", input_shape=(input_size_param, input_size_param, 3)
    )(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    x = tf.keras.layers.Dense(CLASSES, activation="softmax")(x)

    return x
