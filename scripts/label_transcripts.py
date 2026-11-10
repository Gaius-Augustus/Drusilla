#!/usr/bin/env python
"""Project reference CDS onto StringTie-assembled transcripts.

Thin wrapper around ``drusilla.data.label_transcripts.main``. Produces
``labels.npz`` (per-record int8 label arrays) and ``transcripts_labelled.fa``
in --out-dir; feed both into ``build_tfrecords.py`` to create training shards.

Label policy v2 (docs/data_prep.md): unstranded ('.') transcripts are
labelled in both orientations (extra record ``<tid>__rc``), every
non-overlapping complete reference CDS of a transcript is labelled, and
RefSeq ``partial=true`` / ``pseudo=true`` CDS count as incomplete.
``--single-orientation`` restores the forward-only labelling.

Usage::

    python scripts/label_transcripts.py \\
      --stringtie-gtf   path/to/stringtie.gtf \\
      --reference-gff   path/to/refseq.gff \\
      --transcripts-fa  path/to/transcripts.fa \\
      --out-dir         labels/<species>/
"""

from __future__ import annotations

import sys

from drusilla.data.label_transcripts import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
