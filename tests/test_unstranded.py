"""Unstranded StringTie transcripts: split into both orientations, one kept,
projected onto the genome on its own strand. No TF required."""

from __future__ import annotations

import pytest

from drusilla.cli.annotate import (
    _MINUS_SUFFIX,
    _choose_orientations,
    _rev_comp,
    _split_unstranded_gtf,
)
from drusilla.data.label_transcripts import parse_stringtie_gtf
from drusilla.data.tx_to_genome import project_tx_intervals_to_genomic

STOPS = {"TAA", "TAG", "TGA"}


def _gtf_line(tid, strand, kind, start, end):
    return "\t".join([
        "chr1", "StringTie", kind, str(start), str(end), "1000", strand, ".",
        f'gene_id "STRG.1"; transcript_id "{tid}"; cov "5.0";',
    ])


def _write(path, lines):
    path.write_text("# stringtie\n" + "\n".join(lines) + "\n")


def test_split_writes_both_orientations(tmp_path):
    src = tmp_path / "in.gtf"
    _write(src, [
        _gtf_line("STRG.1.1", ".", "transcript", 101, 400),
        _gtf_line("STRG.1.1", ".", "exon", 101, 400),
        _gtf_line("STRG.2.1", "-", "transcript", 501, 900),
        _gtf_line("STRG.2.1", "-", "exon", 501, 600),
        _gtf_line("STRG.2.1", "-", "exon", 701, 900),
    ])
    out = tmp_path / "out.gtf"
    minus_of = _split_unstranded_gtf(src, out)
    minus = "STRG.1.1" + _MINUS_SUFFIX
    assert minus_of == {"STRG.1.1": minus}

    txs = parse_stringtie_gtf(out)
    assert set(txs) == {"STRG.1.1", minus, "STRG.2.1"}
    assert txs["STRG.1.1"].strand == "+"
    assert txs[minus].strand == "-"
    assert txs["STRG.1.1"].exons == txs[minus].exons == [(100, 400)]
    assert txs["STRG.2.1"].strand == "-"
    assert txs["STRG.2.1"].exons == [(500, 600), (700, 900)]
    text = out.read_text()
    assert text.startswith("# stringtie\n")
    assert {ln.split("\t")[6] for ln in text.splitlines()[1:]} == {"+", "-"}
    assert text.count('gene_id "STRG.1"') == 7


def test_split_refuses_existing_minus_id(tmp_path):
    src = tmp_path / "in.gtf"
    _write(src, [
        _gtf_line("t", ".", "exon", 1, 300),
        _gtf_line("t" + _MINUS_SUFFIX, "+", "exon", 1, 300),
    ])
    with pytest.raises(SystemExit):
        _split_unstranded_gtf(src, tmp_path / "out.gtf")


def test_choose_orientations():
    minus_of = {"a": "a-", "b": "b-", "c": "c-", "d": "d-"}
    scores = {
        "a": (300, 0), "a-": (600, 0),     # longer complete ORF on '-'
        "b": (300, 0), "b-": (0, 900),     # a complete ORF beats a partial one
        "c": (0, 0), "c-": (0, 0),         # tie keeps '+'
        "d": (0, 0), "d-": (0, 450),       # only a partial ORF on '-'
    }
    kept = _choose_orientations(minus_of, lambda t: scores[t])
    assert kept == {"a": "a-", "b": "b", "c": "c", "d": "d-"}


def _orf_in(seq):
    """First ATG...stop ORF of seq, half-open."""
    for i in range(len(seq) - 2):
        if seq[i:i + 3] == "ATG":
            for j in range(i, len(seq) - 2, 3):
                if seq[j:j + 3] in STOPS:
                    return i, j + 3
    raise AssertionError("no ORF")


@pytest.mark.parametrize("gene_strand", ["+", "-"])
def test_unstranded_orf_lands_on_its_strand(tmp_path, gene_strand):
    """An ORF called on the extracted copy of a split unstranded transcript
    reads ATG...stop at its projected genomic coordinates on its strand.
    The ORF sits off-centre in the exon, so a mirrored projection (strand '.'
    projected as '-' before the split) would put it elsewhere."""
    orf = "ATG" + "GCC" * 40 + "TAA"
    exon_seq = "C" * 20 + orf + "C" * 200
    if gene_strand == "-":
        exon_seq = _rev_comp(exon_seq)
    genome = "T" * 100 + exon_seq + "T" * 100
    start, end = 101, 100 + len(exon_seq)

    src = tmp_path / "in.gtf"
    _write(src, [_gtf_line("t", ".", "transcript", start, end),
                 _gtf_line("t", ".", "exon", start, end)])
    out = tmp_path / "out.gtf"
    minus_of = _split_unstranded_gtf(src, out)
    txs = parse_stringtie_gtf(out)
    tid = "t" if gene_strand == "+" else minus_of["t"]
    tx = txs[tid]

    # what gffread -w extracts for a stranded transcript
    tx_seq = "".join(genome[s:e] for s, e in tx.exons)
    if tx.strand == "-":
        tx_seq = _rev_comp(tx_seq)
    lines = project_tx_intervals_to_genomic("t", [_orf_in(tx_seq)], tx, "drusilla")

    assert len(lines) == 1
    f = lines[0].split("\t")
    assert f[6] == gene_strand
    assert 'transcript_id "t"' in f[8]
    cds = genome[int(f[3]) - 1:int(f[4])]
    if gene_strand == "-":
        cds = _rev_comp(cds)
    assert cds == orf


def test_projection_refuses_unstranded():
    from drusilla.data.label_transcripts import StringTieTranscript
    tx = StringTieTranscript("t", "chr1", ".", [(1000, 2000)])
    with pytest.raises(ValueError):
        project_tx_intervals_to_genomic("t", [(100, 400)], tx, "drusilla")
