"""Stop codons of the decoder HMM per NCBI translation table. Needs TF."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("hidten")

from drusilla.hmm.annotation_hmm import (  # noqa: E402
    E2, STOP, codon_emissions, DEFAULT_START_CODONS, stop_codons_for_table,
)
from drusilla.hmm.decode import build_decoder_hmm, viterbi_decode_one  # noqa: E402

IR, START, E1 = 0, 1, 2


def test_stop_codons_for_table():
    assert {c for c, _ in stop_codons_for_table(1)} == {"TAA", "TAG", "TGA"}
    assert {c for c, _ in stop_codons_for_table(6)} == {"TGA"}
    assert sum(p for _, p in stop_codons_for_table(1)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stop_codons_for_table(99)


def test_table_6_allows_two_more_sense_codons():
    _, right1 = codon_emissions(DEFAULT_START_CODONS, stop_codons_for_table(1))
    _, right6 = codon_emissions(DEFAULT_START_CODONS, stop_codons_for_table(6))
    assert np.count_nonzero(right1[0, STOP, :64]) == 3
    assert np.count_nonzero(right6[0, STOP, :64]) == 1
    assert np.count_nonzero(right1[0, E2, :64]) == 61
    assert np.count_nonzero(right6[0, E2, :64]) == 63


def _decode(seq: str, table: int) -> np.ndarray:
    """Decode with uninformative logits, so only the codon constraints act
    besides a push towards coding states."""
    L = len(seq)
    nuc = np.zeros((L, 5), dtype=np.float32)
    nuc[np.arange(L), ["ACGT".index(c) for c in seq]] = 1.0
    logits = np.zeros((L, 6), dtype=np.float32)
    logits[:, IR] = -4.0   # prefer an ORF wherever the constraints allow one
    hmm = build_decoder_hmm(parallel=1, translation_table=table)
    return viterbi_decode_one(hmm, logits, nuc)


def test_in_frame_taa_ends_orf_in_table_1_only():
    #      ATG GCT TAA GCT GCT TGA
    seq = "ATGGCTTAAGCTGCTTGA"
    labels1 = _decode(seq, 1)
    labels6 = _decode(seq, 6)
    # table 6: one ORF through the TAA to the TGA
    assert labels6[0] == START and labels6[-1] == STOP
    assert STOP not in labels6[:-1]
    # table 1: no ORF may run through the in-frame TAA
    assert not (labels1[0] == START and labels1[-1] == STOP and STOP not in labels1[:-1])
