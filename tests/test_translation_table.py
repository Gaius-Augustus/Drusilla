"""Frame recoding for genetic codes other than table 1 (data/recode.py)."""

from __future__ import annotations

import pytest

from drusilla.data import recode

# standard code, TCAG order
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
TABLE1 = {a + b + c: _AA[16 * i + 4 * j + k]
          for i, a in enumerate("TCAG") for j, b in enumerate("TCAG") for k, c in enumerate("TCAG")}
# amino acid of the reassigned stop codons (NCBI)
REASSIGNED = {6: "Q", 10: "C", 15: "Q", 29: "Y", 30: "E"}


def test_stop_codons_for_table():
    assert recode.stop_codons_for_table(1) == {"TAA", "TAG", "TGA"}
    assert recode.stop_codons_for_table(6) == {"TGA"}
    assert recode.stop_codons_for_table(10) == {"TAA", "TAG"}
    assert recode.stop_codons_for_table(30) == {"TGA"}
    with pytest.raises(ValueError):
        recode.stop_codons_for_table(99)


@pytest.mark.parametrize("table", sorted(REASSIGNED))
def test_replacement_codon_has_the_amino_acid_of_the_table(table):
    recoding = recode.recoding_for_table(table)
    assert recoding
    for stop, sense in recoding.items():
        assert TABLE1[stop] == "*"
        assert TABLE1[sense] == REASSIGNED[table]


def test_recode_frame_changes_only_the_codons_of_its_frame():
    recoding = recode.recoding_for_table(6)
    #      ATG GCT TAA GCT TAG TGA
    seq = "ATGGCTTAAGCTTAGTGA"
    assert recode.recode_frame(seq, 0, recoding) == "ATGGCTCAAGCTCAGTGA"
    # frame 1: TGG CTT AAG CTT AGT GA; frame 2: GGC TTA AGC TTA GTG A
    assert recode.recode_frame(seq, 1, recoding) == seq
    assert recode.recode_frame(seq, 2, recoding) == seq
    shifted = "C" + seq
    assert recode.recode_frame(shifted, 1, recoding) == "C" + "ATGGCTCAAGCTCAGTGA"
    assert recode.recode_frame(shifted, 0, recoding) == shifted


def test_recode_frame_keeps_length_and_takes_lower_case():
    recoding = recode.recoding_for_table(6)
    for seq in ("", "T", "TA", "taaN", "NNtagtaaT"):
        for frame in range(3):
            assert len(recode.recode_frame(seq, frame, recoding)) == len(seq)
    assert recode.recode_frame("taaN", 0, recoding) == "CAAN"
    assert recode.recode_frame("ATGTAA", 0, {}) == "ATGTAA"


def test_frame_copies_and_ids():
    copies = recode.frame_copies(recode.recoding_for_table(6))("STRG.1.1__rc", "TAATAGA")
    assert [n for n, _ in copies] == ["STRG.1.1__rc__f0", "STRG.1.1__rc__f1", "STRG.1.1__rc__f2"]
    assert [s for _, s in copies] == ["CAACAGA", "TAATAGA", "TAATAGA"]
    assert recode.split_frame_id("STRG.1.1__rc__f2") == ("STRG.1.1__rc", 2)
    with pytest.raises(ValueError):
        recode.split_frame_id("STRG.1.1")

