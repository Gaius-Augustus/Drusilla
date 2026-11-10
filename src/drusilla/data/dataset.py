"""TFRecord dataset loader for Drusilla.

Reads manifests of the form:
    species<TAB>path/to/data.tfrecords
or plain lists of .tfrecords paths (``.tfrecords.gz`` files are read GZIP-
compressed, and the two kinds may be mixed), and returns a tf.data.Dataset of
(input, output, pad_mask) tuples where:
    input:    float32 [L, 6]  — nucleotide one-hot + PAD channel
    output:   float32 [L, 6]  — label one-hot
    pad_mask: bool    [L]     — True where input[..., 5] == 1 (pad position)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def interleave_cycle(n_files: int, cycle_length: int | None = None) -> int:
    """Number of files read side by side: ``cycle_length`` capped at
    ``n_files``; ``None`` keeps the old cap of 32, ``0`` means all files."""
    if n_files <= 0:
        raise ValueError("no TFRecord files")
    if cycle_length is None:
        return min(32, n_files)
    if cycle_length <= 0:
        return n_files
    return min(cycle_length, n_files)


def load_manifest(manifest_path: Path | str) -> list[str]:
    """Return a list of .tfrecords paths from a TSV manifest (species\\tpath)."""
    paths = []
    for line in Path(manifest_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        paths.append(parts[-1])   # last column is always the path
    return paths


def make_dataset(
    sources: Sequence[str] | str | Path,
    chunk_len: int = 9999,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: int = 2048,
    prefetch: int | None = None,
    repeat: bool = False,
    cycle_length: int | None = None,
) -> "tf.data.Dataset":
    """Return a batched tf.data.Dataset from tfrecords paths or a manifest TSV.

    Args:
        sources: a manifest .tsv path, a single .tfrecords path, or a list of
            .tfrecords paths.
        chunk_len: window length used when writing the TFRecords (default 9999).
        batch_size: examples per batch.
        shuffle: whether to shuffle the example stream.
        shuffle_buffer: shuffle buffer size (ignored when shuffle=False).
        prefetch: AUTOTUNE if None.
        repeat: whether to repeat the dataset indefinitely (use for training).
        cycle_length: files interleaved at once (see ``interleave_cycle``);
            None keeps 32, 0 reads all files side by side, which is what a
            442-species manifest needs for every species to be present in
            the shuffle buffer.

    Returns:
        Dataset of (input, output, pad_mask) where each tensor has a leading
        batch dimension. input/output are float32 [B, L, 6]; pad_mask is bool
        [B, L].
    """
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
        )  # [L, 6]
        y = tf.cast(
            tf.io.parse_tensor(parsed["output"], out_type=tf.uint8), tf.float32
        )  # [L, 6]
        # tf.io.parse_tensor returns a fully unknown TensorShape; restore static
        # dims so BiLSTM / Keras internals can call .as_list() without error.
        x.set_shape([chunk_len, 6])
        y.set_shape([chunk_len, 6])
        pad_mask = tf.cast(x[..., 5], tf.bool)  # True where PAD
        return x, y, pad_mask

    def _open(p):
        # Same rule as chunk_tfrecord.tfrecord_compression, on a string tensor.
        compression = tf.where(tf.strings.regex_full_match(p, r".*\.gz"), "GZIP", "")
        return tf.data.TFRecordDataset(p, compression_type=compression)

    # Interleave records from many files simultaneously so the shuffle buffer
    # contains examples from many species at once, not a single sequential file.
    cycle = interleave_cycle(len(file_list), cycle_length)
    file_ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        # Reshuffle file order every pass so each epoch sees a different species order.
        file_ds = file_ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = file_ds.interleave(
        _open,
        cycle_length=cycle,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        # Validation (shuffle=False) must see the same examples in the same
        # order every epoch, in particular under validation_steps.
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    # drop_remainder=True: keep batch dim static so BiLSTM doesn't hit
    # "as_list() is not defined on an unknown TensorShape" in TF 2.17.
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch if prefetch is not None else tf.data.AUTOTUNE)
    return ds
