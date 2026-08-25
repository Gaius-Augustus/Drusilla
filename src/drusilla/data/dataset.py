"""TFRecord dataset loader for Drusilla training.

Reads manifests of the form:
    species<TAB>path/to/data.tfrecords
or plain lists of .tfrecords paths, and returns a tf.data.Dataset of
(input, output, pad_mask) tuples where:
    input:    float32 [L, 6]  - nucleotide one-hot + PAD channel
    output:   float32 [L, 6]  - label one-hot
    pad_mask: bool    [L]     - True where input[..., 5] == 1 (pad position)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def load_manifest(manifest_path: Path | str) -> list[str]:
    """Return a list of .tfrecords paths from a TSV manifest (species\\tpath)."""
    paths = []
    for line in Path(manifest_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        paths.append(parts[-1])
    return paths


def make_dataset(
    sources: Sequence[str] | str | Path,
    chunk_len: int = 9999,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: int = 2048,
    prefetch: int | None = None,
    repeat: bool = False,
) -> "tf.data.Dataset":
    """Return a batched tf.data.Dataset from tfrecords paths or a manifest TSV."""
    import tensorflow as tf

    if isinstance(sources, (str, Path)):
        src = str(sources)
        if src.endswith(".tsv") or src.endswith(".txt"):
            file_list = load_manifest(src)
        else:
            file_list = [src]
    else:
        file_list = [str(p) for p in sources]

    spec = {
        "input":     tf.io.FixedLenFeature([], tf.string),
        "output":    tf.io.FixedLenFeature([], tf.string),
        "tx_id":     tf.io.FixedLenFeature([], tf.string),
        "chunk_idx": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse(serialized):
        parsed = tf.io.parse_single_example(serialized, spec)
        x = tf.cast(
            tf.io.parse_tensor(parsed["input"],  out_type=tf.uint8), tf.float32
        )
        y = tf.cast(
            tf.io.parse_tensor(parsed["output"], out_type=tf.uint8), tf.float32
        )
        # tf.io.parse_tensor returns a fully unknown TensorShape; restore
        # static dims so BiLSTM / Keras internals can call .as_list().
        x.set_shape([chunk_len, 6])
        y.set_shape([chunk_len, 6])
        pad_mask = tf.cast(x[..., 5], tf.bool)
        return x, y, pad_mask

    cycle = min(32, len(file_list))
    file_ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        file_ds = file_ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = file_ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=""),
        cycle_length=cycle,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    # drop_remainder=True keeps batch dim static (BiLSTM needs it in TF 2.17).
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch if prefetch is not None else tf.data.AUTOTUNE)
    return ds
