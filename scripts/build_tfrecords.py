#!/usr/bin/env python
"""Chunk labelled transcripts into fixed-length windows and write TFRecords.

Thin wrapper around ``drusilla.data.chunk_tfrecord.main``. Input is the
labelled FASTA + labels npz produced by ``label_transcripts.py``.

Usage::

    python scripts/build_tfrecords.py \\
      --fasta   labels/<species>/transcripts_labelled.fa \\
      --labels  labels/<species>/labels.npz \\
      --out     tfrecords/<species>.tfrecords \\
      --chunk-len 9999
"""

from __future__ import annotations

import sys

from drusilla.data.chunk_tfrecord import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
