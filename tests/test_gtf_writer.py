"""ORF extraction and GTF projection tests. Pure numpy — no TF required."""

from __future__ import annotations

import numpy as np

from drusilla.data.gtf_writer import (
    extract_orfs,
    extract_partial_orfs,
    extract_5prime_partial_orfs,
    classify_lorf,
)


IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5


def test_extract_complete_orf():
    # IR ... START E1 E2 E0 E1 STOP ... IR
    labels = np.array([IR, IR, START, E1, E2, E0, E1, STOP, IR, IR])
    assert extract_orfs(labels) == [(2, 8)]


def test_no_orf_without_stop():
    labels = np.array([IR, START, E1, E2, E0, E1, IR, IR])
    assert extract_orfs(labels) == []


def test_extract_partial_orf():
    # 3'-truncated: START ... but no STOP before end
    labels = np.array([IR, START, E1, E2, E0])
    assert extract_partial_orfs(labels) == [(1, 5)]


def test_extract_5prime_partial():
    # 5'-truncated: coding at position 0, no upstream START, ends at STOP
    labels = np.array([E1, E2, E0, E1, STOP, IR])
    assert extract_5prime_partial_orfs(labels) == [(0, 5)]


def test_5prime_partial_needs_stop():
    labels = np.array([E1, E2, E0, IR])
    assert extract_5prime_partial_orfs(labels) == []


def test_classify_lorf_no_upstream():
    # LORF_NOUPSTOP: no upstream stop and no upstream ATG.
    seq = "AAA" * 5 + "ATG" + "AAA" * 3
    cls = classify_lorf(seq, orf_tx_start=15)
    assert cls == "LORF_NOUPSTOP"


def test_classify_lorf_upstop():
    # LORF_UPSTOP: upstream in-frame TAA, no ATG between it and our ATG.
    seq = "TAA" + "CCC" * 3 + "ATG"
    cls = classify_lorf(seq, orf_tx_start=12)
    assert cls == "LORF_UPSTOP"


def test_classify_up_lorf():
    # upLORF: no upstream stop; an upstream in-frame ATG exists.
    seq = "ATG" + "CCC" * 3 + "ATG"
    cls = classify_lorf(seq, orf_tx_start=12)
    assert cls == "upLORF"
