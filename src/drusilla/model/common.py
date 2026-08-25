"""Shared building blocks used by all ORF model variants."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


NUM_INPUT_CHANNELS = 6
NUM_CLASSES = 6


def zero_pad_channel(inp):
    """Mask out the PAD channel (index 5) so padded positions carry no signal."""
    return keras.layers.Lambda(
        lambda t: t * tf.concat(
            [tf.ones_like(t[..., :5]), tf.zeros_like(t[..., 5:])], axis=-1
        ),
        name="zero_pad_channel",
    )(inp)


def conv_stem(x, filters: int, kernel: int, n_layers: int):
    """N x Conv1D(filters, kernel, padding=same, relu)."""
    for i in range(n_layers):
        x = keras.layers.Conv1D(
            filters=filters, kernel_size=kernel,
            padding="same", activation="relu",
            name=f"conv_{i+1}",
        )(x)
    return x
