"""Tests for the subseq-collapse post-step. Pure Python."""

from __future__ import annotations

from pathlib import Path

import pytest

from drusilla.postprocess.subseq_filter import (
    gene_id_from_tx,
    is_subsequence,
    find_dropable,
    run_filter,
)


def test_gene_id_from_tx_stringtie():
    assert gene_id_from_tx("STRG.42.3") == "STRG.42"
    assert gene_id_from_tx("STRG.1.1") == "STRG.1"


def test_gene_id_from_tx_falls_back():
    # No trailing .<int> -> return as-is
    assert gene_id_from_tx("ENSG00000012345") == "ENSG00000012345"


# ---------- is_subsequence ----------

def test_single_exon_contained():
    a = [(100, 200)]
    b = [(50, 300)]
    assert is_subsequence(a, b) is True


def test_single_exon_not_contained():
    a = [(50, 200)]
    b = [(100, 300)]
    assert is_subsequence(a, b) is False


def test_two_exon_exact_match_first_pair():
    # A: [50-100), [200-300)  ; B: [50-100), [200-300), [400-500)
    # A's leftmost ends at B[0][1]=100 (splice donor match), start >= 50 -> OK
    # A's rightmost starts at B[1][0]=200 (splice acceptor match), end <= 300 -> OK
    a = [(50, 100), (200, 300)]
    b = [(50, 100), (200, 300), (400, 500)]
    assert is_subsequence(a, b) is True


def test_three_exon_interior_must_match_exact():
    # Interior block differs from any B interior -> not a subseq (strict).
    a = [(50, 100), (150, 250), (300, 400)]
    b = [(50, 100), (200, 250), (300, 400)]
    assert is_subsequence(a, b) is False


def test_terminal_overhang_tolerance():
    # A's leftmost extends 5nt earlier than B's, but shares splice donor.
    a = [(45, 100), (200, 300)]
    b = [(50, 100), (200, 300)]
    assert is_subsequence(a, b, terminal_overhang_nt=10) is True
    assert is_subsequence(a, b, terminal_overhang_nt=0)  is False


def test_splice_shift_must_be_frame_preserving():
    # Interior boundary shifted by 6nt (multiple of 3) -> OK if shift>=6.
    a = [(50, 100), (206, 250), (300, 400)]
    b = [(50, 100), (200, 250), (300, 400)]
    assert is_subsequence(a, b, splice_shift_nt=6) is True
    # Shifted by 4nt -> not a multiple of 3 -> rejected.
    a2 = [(50, 100), (204, 250), (300, 400)]
    assert is_subsequence(a2, b, splice_shift_nt=6) is False


def test_a_longer_than_b_never_subseq():
    a = [(0, 100), (200, 300), (400, 500)]
    b = [(0, 100), (200, 300)]
    assert is_subsequence(a, b) is False


def test_allow_exon_skip():
    # A skips B's middle exon.
    a = [(50, 100), (400, 500)]  # matches B[0] and B[2]
    b = [(50, 100), (200, 300), (400, 500)]
    # Strict (no skip): A's leftmost matches B[0] (a[1]==b[0][1]=100),
    # A's rightmost with a[0]=400 must match b_slice[-1][0]. For k=0,
    # b_slice=[b[0],b[1]] and b_slice[-1]=(200,300), no match. For k=1,
    # b_slice=[b[1],b[2]] and b_slice[0]=(200,300), a[0]=50 != b_slice[0][1]=300.
    # So strict is False.
    assert is_subsequence(a, b, allow_exon_skip=False) is False
    assert is_subsequence(a, b, allow_exon_skip=True) is True


# ---------- find_dropable ----------

def _rec(contig, strand, exons, lines=None):
    return {
        "contig": contig,
        "strand": strand,
        "exons": sorted(exons),
        "lines": lines or [],
    }


def test_find_dropable_drops_short_isoform():
    # STRG.7.1 is single-exon and fully inside STRG.7.2 -> drop 7.1.
    by_tid = {
        "STRG.7.1": _rec("chr1", "+", [(100, 200)]),
        "STRG.7.2": _rec("chr1", "+", [(50, 300)]),
    }
    drop = find_dropable(by_tid)
    assert drop == {"STRG.7.1": "STRG.7.2"}


def test_find_dropable_ignores_different_locus():
    by_tid = {
        "STRG.7.1": _rec("chr1", "+", [(100, 200)]),
        "STRG.8.1": _rec("chr1", "+", [(50, 300)]),
    }
    # Different loci (STRG.7 vs STRG.8) -> no drops even though contained.
    assert find_dropable(by_tid) == {}


def test_find_dropable_ignores_opposite_strand():
    by_tid = {
        "STRG.7.1": _rec("chr1", "+", [(100, 200)]),
        "STRG.7.2": _rec("chr1", "-", [(50, 300)]),
    }
    assert find_dropable(by_tid) == {}


def test_find_dropable_reroot_chain():
    # If A is subseq of B and B is subseq of C, both A and B should
    # ultimately point at C in the report.
    by_tid = {
        "STRG.1.1": _rec("chr1", "+", [(100, 150)]),          # A
        "STRG.1.2": _rec("chr1", "+", [(50, 200)]),           # B (contains A)
        "STRG.1.3": _rec("chr1", "+", [(0, 250), (400, 500)]),  # C longer
    }
    drop = find_dropable(by_tid)
    # 1.1 (single-exon) is inside 1.2 and inside 1.3's first exon.
    # 1.2 (single-exon) is inside 1.3's first exon.
    # Sorted by (-len(exons), -span, tid): 1.3 (2 exons) first, then 1.2 and 1.1 by span.
    # Every kept-vs-A check will pick the FIRST keeper found in iteration order.
    assert "STRG.1.1" in drop
    assert "STRG.1.2" in drop
    # After re-rooting, both keepers should be a not-dropped tid.
    for keeper in drop.values():
        assert keeper not in drop


# ---------- run_filter (end-to-end small GTF) ----------

def _write_gtf(path: Path, tid_to_exons: dict[str, list[tuple[int, int]]],
               contig: str = "chr1", strand: str = "+") -> None:
    lines = []
    for tid, exons in tid_to_exons.items():
        for s, e in exons:
            attrs = f'transcript_id "{tid}"; gene_id "{tid}";'
            lines.append("\t".join([
                contig, "drusilla", "CDS",
                str(s + 1), str(e), ".", strand, "0", attrs,
            ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_filter_drops_and_writes(tmp_path: Path):
    gtf = tmp_path / "orfs.gtf"
    out = tmp_path / "orfs.filtered.gtf"
    rep = tmp_path / "dropped.tsv"
    _write_gtf(gtf, {
        "STRG.1.1": [(100, 200)],
        "STRG.1.2": [(50, 300)],
        "STRG.2.1": [(1000, 1200), (1500, 1800)],
    })
    stats = run_filter(gtf, out, rep)
    assert stats["n_input_tx"] == 3
    assert stats["n_dropped_tx"] == 1
    assert stats["n_kept_tx"] == 2
    kept_lines = out.read_text().strip().splitlines()
    kept_tids = {ln.split("transcript_id \"", 1)[1].split("\"", 1)[0]
                 for ln in kept_lines}
    assert "STRG.1.1" not in kept_tids
    assert "STRG.1.2" in kept_tids
    assert "STRG.2.1" in kept_tids
    rep_lines = rep.read_text().strip().splitlines()
    assert rep_lines[0] == "dropped_tid\tkeeper_tid\treason"
    assert any(ln.startswith("STRG.1.1\tSTRG.1.2\t") for ln in rep_lines[1:])


def test_run_filter_negative_tolerance_rejected(tmp_path: Path):
    gtf = tmp_path / "orfs.gtf"
    _write_gtf(gtf, {"STRG.1.1": [(0, 100)]})
    with pytest.raises(ValueError):
        run_filter(gtf, tmp_path / "out.gtf",
                   terminal_overhang_nt=-1)
