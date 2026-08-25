"""CNN + reshape-pooled BiLSTM ORF model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .common import NUM_INPUT_CHANNELS, NUM_CLASSES, zero_pad_channel, conv_stem


def build_cnn_lstm(
    chunk_len: int = 9999,
    conv_filters: int = 64,
    conv_kernel: int = 9,
    conv_layers: int = 2,
    pool_size: int = 9,
    lstm_units: int = 200,
    lstm_layers: int = 2,
    dropout: float = 0.1,
    head_hidden: int = 32,
) -> keras.Model:
    if chunk_len % pool_size != 0:
        raise ValueError(f"chunk_len ({chunk_len}) must be divisible by pool_size ({pool_size})")
    L_reduced = chunk_len // pool_size

    inp = keras.Input(shape=(chunk_len, NUM_INPUT_CHANNELS), name="input")
    x = zero_pad_channel(inp)
    x = conv_stem(x, conv_filters, conv_kernel, conv_layers)

    x = keras.layers.Reshape((L_reduced, pool_size * conv_filters), name="reshape_pool")(x)

    for i in range(lstm_layers):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units, return_sequences=True,
                              dropout=dropout, recurrent_dropout=0.0),
            name=f"bilstm_{i+1}",
        )(x)
        if dropout > 0.0:
            x = keras.layers.Dropout(dropout, name=f"drop_{i+1}")(x)

    x = keras.layers.Dense(pool_size * head_hidden, activation="relu", name="post_lstm_dense")(x)
    x = keras.layers.Reshape((chunk_len, head_hidden), name="reshape_unpool")(x)
    logits = keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits, name="drusilla_cnn_lstm")
