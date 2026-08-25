"""Chunk labelled transcripts into fixed-length windows and write TFRecords.

Layout per chunk:
  * input  shape [L, 6]  channels = A, C, G, T, N, PAD
  * output shape [L, 6]  classes  = IR, START, E1, E2, E0, STOP

Convention for padded positions (where the transcript is shorter than L or the
final chunk is partial): input has PAD=1 and all other channels 0; output is
all-zero (no class is one-hot). The training loop must ignore positions where
`input[..., 5] == 1` when computing the loss.

TFRecord schema:
  * `input`    bytes - serialized uint8 tensor [L, 6]
  * `output`   bytes - serialized uint8 tensor [L, 6]
  * `tx_id`    bytes - utf-8 transcript id
  * `chunk_idx` int64 - 0-based chunk index within the transcript
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


DEFAULT_CHUNK_LEN = 9999
NUM_NT_CHANNELS = 5
NUM_INPUT_CHANNELS = 6
NUM_CLASSES = 6


_NT_TO_IDX = {ord(c): i for c, i in zip("ACGTN", range(5))}
_NT_TO_IDX.update({ord(c.lower()): i for c, i in zip("ACGTN", range(5))})


def read_fasta(path: Path | str) -> dict[str, str]:
    """Read a FASTA file into a ``{record_id: sequence}`` dict."""
    out: dict[str, str] = {}
    current_name: str | None = None
    current_chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                out[current_name] = "".join(current_chunks)
            current_name = line[1:].strip().split()[0]
            current_chunks = []
        else:
            current_chunks.append(line.strip())
    if current_name is not None:
        out[current_name] = "".join(current_chunks)
    return out


def encode_nucleotides(seq: str) -> np.ndarray:
    """Return a [L, 5] uint8 one-hot of A,C,G,T,N (any other letter -> N)."""
    arr = np.zeros((len(seq), NUM_NT_CHANNELS), dtype=np.uint8)
    for i, ch in enumerate(seq):
        arr[i, _NT_TO_IDX.get(ord(ch), 4)] = 1
    return arr


def labels_to_onehot(labels: np.ndarray) -> np.ndarray:
    """Turn a (L,) int label array into a (L, 6) uint8 one-hot."""
    if labels.ndim != 1:
        raise ValueError(f"expected 1-D labels, got shape {labels.shape}")
    out = np.zeros((labels.shape[0], NUM_CLASSES), dtype=np.uint8)
    out[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1
    return out


def pad_to_length(arr: np.ndarray, length: int, fill: int = 0) -> np.ndarray:
    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full((length,), fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def chunk_transcript(
    seq: str,
    labels: np.ndarray,
    chunk_len: int = DEFAULT_CHUNK_LEN,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(input_chunk, output_chunk)`` pairs for one transcript."""
    if len(seq) != labels.shape[0]:
        raise ValueError(
            f"seq length {len(seq)} does not match labels length {labels.shape[0]}"
        )
    if len(seq) == 0:
        return

    n_chunks = (len(seq) + chunk_len - 1) // chunk_len
    for ci in range(n_chunks):
        start = ci * chunk_len
        end = start + chunk_len
        sub_seq = seq[start:end]
        sub_lab = labels[start:end]
        real_len = len(sub_seq)

        input_chunk = np.zeros((chunk_len, NUM_INPUT_CHANNELS), dtype=np.uint8)
        if real_len > 0:
            input_chunk[:real_len, :NUM_NT_CHANNELS] = encode_nucleotides(sub_seq)
        input_chunk[real_len:, NUM_NT_CHANNELS] = 1

        output_chunk = np.zeros((chunk_len, NUM_CLASSES), dtype=np.uint8)
        if real_len > 0:
            output_chunk[:real_len] = labels_to_onehot(np.asarray(sub_lab))

        yield input_chunk, output_chunk


def chunk_all(
    sequences: dict[str, str],
    labels: dict[str, np.ndarray],
    chunk_len: int = DEFAULT_CHUNK_LEN,
) -> Iterator[tuple[str, int, np.ndarray, np.ndarray]]:
    """Iterate ``(tx_id, chunk_idx, input, output)`` across a full transcript set."""
    common = sorted(set(sequences) & set(labels))
    for tid in common:
        seq = sequences[tid]
        lab = labels[tid]
        for ci, (x, y) in enumerate(chunk_transcript(seq, lab, chunk_len)):
            yield tid, ci, x, y


def _tf():
    import tensorflow as tf
    return tf


def serialize_example(
    input_chunk: np.ndarray,
    output_chunk: np.ndarray,
    tx_id: str,
    chunk_idx: int,
) -> bytes:
    tf = _tf()
    if input_chunk.dtype != np.uint8 or output_chunk.dtype != np.uint8:
        raise TypeError("expected uint8 chunks for TFRecord serialization")

    ex = tf.train.Example(features=tf.train.Features(feature={
        "input": tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(input_chunk).numpy()])),
        "output": tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(output_chunk).numpy()])),
        "tx_id": tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tx_id.encode("utf-8")])),
        "chunk_idx": tf.train.Feature(int64_list=tf.train.Int64List(
            value=[int(chunk_idx)])),
    }))
    return ex.SerializeToString()


def write_tfrecord(
    path: Path | str,
    examples: Iterable[tuple[str, int, np.ndarray, np.ndarray]],
) -> int:
    tf = _tf()
    n = 0
    with tf.io.TFRecordWriter(str(path)) as writer:
        for tx_id, chunk_idx, x, y in examples:
            writer.write(serialize_example(x, y, tx_id, chunk_idx))
            n += 1
    return n


def parse_example_spec(chunk_len: int = DEFAULT_CHUNK_LEN):
    tf = _tf()
    return {
        "input":     tf.io.FixedLenFeature([], tf.string),
        "output":    tf.io.FixedLenFeature([], tf.string),
        "tx_id":     tf.io.FixedLenFeature([], tf.string),
        "chunk_idx": tf.io.FixedLenFeature([], tf.int64),
    }


def decode_example(serialized: bytes, chunk_len: int = DEFAULT_CHUNK_LEN):
    tf = _tf()
    parsed = tf.io.parse_single_example(serialized, parse_example_spec(chunk_len))
    x = tf.io.parse_tensor(parsed["input"], out_type=tf.uint8).numpy()
    y = tf.io.parse_tensor(parsed["output"], out_type=tf.uint8).numpy()
    tx_id = parsed["tx_id"].numpy().decode("utf-8")
    chunk_idx = int(parsed["chunk_idx"].numpy())
    return x, y, tx_id, chunk_idx


def main(argv: list[str] | None = None) -> int:
    """Chunk a labelled transcripts FASTA + labels npz into a TFRecord shard."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", type=Path, required=True,
                    help="transcripts_labelled.fa")
    ap.add_argument("--labels", type=Path, required=True,
                    help="labels.npz (keys = transcript IDs)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .tfrecords path")
    ap.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    args = ap.parse_args(argv)

    sequences = read_fasta(args.fasta)
    with np.load(args.labels, allow_pickle=False) as npz:
        labels = {k: np.asarray(npz[k]) for k in npz.files}

    n = write_tfrecord(args.out,
                       chunk_all(sequences, labels, chunk_len=args.chunk_len))
    print(f"wrote {n} chunks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
