"""Tests for the pure-numpy chunking/encoding layer. TFRecord I/O is covered
only if tensorflow is importable in the current environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drusilla.data.chunk_tfrecord import (
    DEFAULT_CHUNK_LEN,
    NUM_CLASSES,
    NUM_INPUT_CHANNELS,
    chunk_all,
    chunk_transcript,
    encode_nucleotides,
    labels_to_onehot,
    read_fasta,
)
from drusilla.data.label_transcripts import E0, E1, E2, IR, START, STOP


def test_encode_nucleotides_onehot():
    arr = encode_nucleotides("AcGtNx")
    expected = np.array([
        [1, 0, 0, 0, 0],   # A
        [0, 1, 0, 0, 0],   # c -> C
        [0, 0, 1, 0, 0],   # G
        [0, 0, 0, 1, 0],   # t -> T
        [0, 0, 0, 0, 1],   # N
        [0, 0, 0, 0, 1],   # unknown -> N
    ], dtype=np.uint8)
    assert np.array_equal(arr, expected)


def test_labels_to_onehot_round_trip():
    labels = np.array([IR, START, E1, E2, E0, STOP, IR], dtype=np.int8)
    oh = labels_to_onehot(labels)
    assert oh.shape == (7, NUM_CLASSES)
    assert np.array_equal(np.argmax(oh, axis=1), labels.astype(np.int64))


def test_chunk_shorter_than_chunk_len_pads():
    seq = "ACGT"
    labels = np.array([IR, START, E1, E2], dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=8))
    assert len(chunks) == 1

    x, y = chunks[0]
    assert x.shape == (8, NUM_INPUT_CHANNELS)
    assert y.shape == (8, NUM_CLASSES)

    # First 4 positions: real nt one-hot, no pad.
    assert (x[:4, :5].sum(axis=1) == 1).all()
    assert (x[:4, 5] == 0).all()
    # Pad tail: pad channel = 1, all other channels = 0.
    assert (x[4:, 5] == 1).all()
    assert (x[4:, :5] == 0).all()

    # Labels: first 4 are one-hot of IR/START/E1/E2, rest are all zero.
    assert np.array_equal(np.argmax(y[:4], axis=1), [IR, START, E1, E2])
    assert (y[4:] == 0).all()


def test_chunk_exact_multiple_no_padding():
    L = 4
    seq = "ACGTACGT"      # length 8 = 2 * L
    labels = np.zeros(8, dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=L))
    assert len(chunks) == 2
    for x, y in chunks:
        assert x.shape == (L, NUM_INPUT_CHANNELS)
        assert (x[:, 5] == 0).all()        # no pad anywhere


def test_chunk_longer_than_chunk_len_pads_tail():
    L = 4
    seq = "ACGTAC"        # length 6 -> two chunks: 4 nt + 2 nt (padded)
    labels = np.arange(6, dtype=np.int8) % NUM_CLASSES
    chunks = list(chunk_transcript(seq, labels, chunk_len=L))
    assert len(chunks) == 2

    x2, y2 = chunks[1]
    assert x2.shape == (L, NUM_INPUT_CHANNELS)
    assert (x2[:2, 5] == 0).all()          # real positions
    assert (x2[2:, 5] == 1).all()          # padded tail
    assert (y2[2:] == 0).all()             # all-zero label on pad


def test_chunk_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        list(chunk_transcript("ACGT", np.zeros(3, dtype=np.int8)))


def test_chunk_all_iterates_in_deterministic_order():
    sequences = {"T2": "ACGT", "T1": "GGG"}
    labels = {"T1": np.zeros(3, dtype=np.int8),
              "T2": np.zeros(4, dtype=np.int8)}
    chunks = list(chunk_all(sequences, labels, chunk_len=4))
    # Sorted by tx_id -> T1, T2.
    assert [tid for tid, _, _, _ in chunks] == ["T1", "T2"]


def test_read_fasta(tmp_path):
    p = tmp_path / "x.fa"
    p.write_text(">a extra header\nACGT\nACGT\n>b\nNNN\n", encoding="utf-8")
    seqs = read_fasta(p)
    assert seqs == {"a": "ACGTACGT", "b": "NNN"}


def test_tfrecord_compression_follows_the_suffix():
    from drusilla.data.chunk_tfrecord import tfrecord_compression
    assert tfrecord_compression("x/data.tfrecords") == ""
    assert tfrecord_compression("x/data.tfrecords.gz") == "GZIP"
    # The rechunk job writes to <out>.tmp first and renames afterwards.
    assert tfrecord_compression(Path("x/data.tfrecords.gz.tmp")) == "GZIP"
    assert tfrecord_compression("x/data.tfrecords.tmp") == ""


# ---------- optional TFRecord tests (skip if tensorflow is unavailable) ----------

tf_available = False
try:
    import tensorflow as tf  # noqa: F401
    tf_available = True
except Exception:
    pass


@pytest.mark.skipif(not tf_available, reason="tensorflow not installed")
def test_tfrecord_round_trip(tmp_path):
    from drusilla.data.chunk_tfrecord import (
        decode_example,
        write_tfrecord,
    )
    seq = "ACGTACGTAC"
    labels = np.array([IR, START, E1, E2, E0, E1, E2, E0, E1, STOP],
                      dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=DEFAULT_CHUNK_LEN))
    examples = [("tx1", ci, x, y) for ci, (x, y) in enumerate(chunks)]

    path = tmp_path / "out.tfrecords"
    n = write_tfrecord(path, examples)
    assert n == len(examples)

    import tensorflow as tf
    ds = tf.data.TFRecordDataset([str(path)])
    decoded = [decode_example(rec.numpy()) for rec in ds]
    assert len(decoded) == n
    x0, y0, tid0, ci0 = decoded[0]
    assert tid0 == "tx1"
    assert ci0 == 0
    assert x0.shape == (DEFAULT_CHUNK_LEN, NUM_INPUT_CHANNELS)
    assert y0.shape == (DEFAULT_CHUNK_LEN, NUM_CLASSES)
    # First 10 positions should match our input sequence/labels.
    assert np.array_equal(x0[:10, :5], encode_nucleotides(seq))
    assert np.array_equal(np.argmax(y0[:10], axis=1), labels)


@pytest.mark.skipif(not tf_available, reason="tensorflow not installed")
def test_gzip_tfrecord_round_trip_through_make_dataset(tmp_path):
    import gzip
    import tensorflow as tf
    from drusilla.data.chunk_tfrecord import decode_example, write_tfrecord
    from drusilla.data.dataset import make_dataset

    L = 16
    seq = "ACGTACGTAC"
    labels = np.array([IR, START, E1, E2, E0, E1, E2, E0, E1, STOP], dtype=np.int8)
    examples = [("tx1", ci, x, y)
                for ci, (x, y) in enumerate(chunk_transcript(seq, labels, chunk_len=L))]

    plain = tmp_path / "plain.tfrecords"
    packed = tmp_path / "packed.tfrecords.gz.tmp"     # as the rechunk job names it
    write_tfrecord(plain, examples)
    write_tfrecord(packed, examples)
    packed = packed.rename(tmp_path / "packed.tfrecords.gz")
    with open(packed, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"
    with gzip.open(packed, "rb") as fh, open(plain, "rb") as ref:
        assert fh.read() == ref.read()

    ds = tf.data.TFRecordDataset([str(packed)], compression_type="GZIP")
    x0, _, tid0, _ = decode_example(next(iter(ds)).numpy(), chunk_len=L)
    assert tid0 == "tx1"
    assert np.array_equal(x0[:10, :5], encode_nucleotides(seq))

    # make_dataset picks the compression per file, so both kinds can be mixed.
    mixed = make_dataset([str(plain), str(packed)], chunk_len=L, batch_size=2,
                         shuffle=False)
    xb, yb, mask = next(iter(mixed))
    assert xb.shape == (2, L, NUM_INPUT_CHANNELS)
    assert yb.shape == (2, L, NUM_CLASSES)
    assert bool(mask[0, L - 1]) and not bool(mask[0, 0])


def test_interleave_cycle_caps_and_zero_means_all():
    from drusilla.data.dataset import interleave_cycle
    assert interleave_cycle(442) == 32            # old default kept
    assert interleave_cycle(10) == 10
    assert interleave_cycle(442, 0) == 442        # 0 = every file side by side
    assert interleave_cycle(442, 128) == 128
    assert interleave_cycle(6, 128) == 6
    with pytest.raises(ValueError):
        interleave_cycle(0)
