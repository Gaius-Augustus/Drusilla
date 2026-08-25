"""CNN + reshape-pooled Transformer-encoder ORF model.

Same conv stem and ÷pool_size reshape as cnn_lstm; the BiLSTM stack is
replaced by N transformer-encoder blocks (multi-head self-attention + FFN
with pre-LayerNorm, residual connections). Learned positional embeddings
are added to the reduced-length sequence.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .common import NUM_INPUT_CHANNELS, NUM_CLASSES, zero_pad_channel, conv_stem


def _transformer_block(x, d_model: int, num_heads: int, ff_dim: int,
                       dropout: float, idx: int):
    h = keras.layers.LayerNormalization(name=f"tr_norm1_{idx}")(x)
    h = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout, name=f"tr_mha_{idx}",
    )(h, h)
    x = keras.layers.Add(name=f"tr_add1_{idx}")([x, h])

    h = keras.layers.LayerNormalization(name=f"tr_norm2_{idx}")(x)
    h = keras.layers.Dense(ff_dim, activation="gelu", name=f"tr_ff1_{idx}")(h)
    h = keras.layers.Dense(d_model, name=f"tr_ff2_{idx}")(h)
    h = keras.layers.Dropout(dropout, name=f"tr_drop_{idx}")(h)
    return keras.layers.Add(name=f"tr_add2_{idx}")([x, h])


def build_cnn_transformer(
    chunk_len: int = 9999,
    conv_filters: int = 64,
    conv_kernel: int = 9,
    conv_layers: int = 2,
    pool_size: int = 9,
    d_model: int = 256,
    num_heads: int = 4,
    ff_dim: int = 512,
    transformer_layers: int = 4,
    dropout: float = 0.1,
    head_hidden: int = 32,
) -> keras.Model:
    if chunk_len % pool_size != 0:
        raise ValueError(f"chunk_len ({chunk_len}) must be divisible by pool_size ({pool_size})")
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    L_reduced = chunk_len // pool_size

    inp = keras.Input(shape=(chunk_len, NUM_INPUT_CHANNELS), name="input")
    x = zero_pad_channel(inp)
    x = conv_stem(x, conv_filters, conv_kernel, conv_layers)

    x = keras.layers.Reshape((L_reduced, pool_size * conv_filters), name="reshape_pool")(x)
    x = keras.layers.Dense(d_model, name="proj_to_dmodel")(x)

    pos = keras.layers.Embedding(L_reduced, d_model, name="pos_embed")(
        tf.range(L_reduced)[tf.newaxis, :]
    )
    x = keras.layers.Add(name="add_pos")([x, pos])

    for i in range(transformer_layers):
        x = _transformer_block(x, d_model, num_heads, ff_dim, dropout, idx=i + 1)

    x = keras.layers.LayerNormalization(name="tr_norm_final")(x)
    x = keras.layers.Dense(pool_size * head_hidden, activation="relu",
                           name="post_tr_dense")(x)
    x = keras.layers.Reshape((chunk_len, head_hidden), name="reshape_unpool")(x)
    logits = keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits, name="drusilla_cnn_transformer")
