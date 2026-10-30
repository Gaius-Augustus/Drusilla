"""Unit tests for transcript augmentation (unstranded copies, genomic flanks,
ORF placement, multi-ORF ids)."""

from __future__ import annotations

from drusilla.data.label_transcripts import StringTieTranscript, parse_stringtie_gtf
from drusilla.data.tx_augment import (
    DictGenome,
    RC_SUFFIX,
    assign_orf_ids,
    augment_unstranded,
    base_tid,
    build_flank,
    clip_5p,
    iter_fasta,
    n_pad_only,
    pick_orientation,
    place_orf,
    rev_comp,
    rewrite_fasta,
    with_rc_copies,
)
from drusilla.data.tx_to_genome import project_tx_intervals_to_genomic


def _tx(tid="T1", contig="c1", strand="+", exons=((10, 20),), unstranded=False):
    return StringTieTranscript(tid=tid, contig=contig, strand=strand,
                               exons=[tuple(e) for e in exons], unstranded=unstranded)


# ---------- ids ----------

def test_base_tid_strips_suffixes():
    assert base_tid("STRG.1.1") == "STRG.1.1"
    assert base_tid("STRG.1.1__rc") == "STRG.1.1"
    assert base_tid("STRG.1.1.orf2") == "STRG.1.1"
    assert base_tid("STRG.1.1__rc.orf3") == "STRG.1.1"
    # a trailing ".orf" without number is not a suffix
    assert base_tid("gene.orf") == "gene.orf"


def test_assign_orf_ids():
    assert assign_orf_ids("T", 1) == ["T"]
    assert assign_orf_ids("T", 3) == ["T", "T.orf2", "T.orf3"]


# ---------- unstranded ----------

def test_parse_normalises_dot_strand(tmp_path):
    gtf = tmp_path / "st.gtf"
    gtf.write_text('c1\tStringTie\texon\t11\t20\t.\t.\t.\ttranscript_id "U1";\n'
                   'c1\tStringTie\texon\t31\t40\t.\t-\t.\ttranscript_id "M1";\n')
    st = parse_stringtie_gtf(gtf)
    assert st["U1"].strand == "+" and st["U1"].unstranded
    assert st["M1"].strand == "-" and not st["M1"].unstranded


def test_augment_unstranded_adds_rc_copy():
    txs = {"U1": _tx("U1", strand="+", unstranded=True), "P1": _tx("P1")}
    out, rc = augment_unstranded(txs)
    assert rc == ["U1"]
    assert set(out) == {"U1", "U1" + RC_SUFFIX, "P1"}
    assert out["U1"].strand == "+" and out["U1__rc"].strand == "-"
    assert out["U1__rc"].exons == out["U1"].exons
    assert out["P1"] is txs["P1"]


def test_rc_copy_projects_to_minus_strand_coordinates():
    # forward copy: ORF at tx [2, 8) on exon [10, 20) -> genomic [12, 18) -> 1-based 13..18
    fwd = _tx("U1", strand="+")
    rc = _tx("U1__rc", strand="-")
    f = project_tx_intervals_to_genomic("U1", [(2, 8)], fwd, "x")
    r = project_tx_intervals_to_genomic("U1__rc", [(2, 8)], rc, "x")
    assert f[0].split("\t")[3:5] == ["13", "18"] and f[0].split("\t")[6] == "+"
    # same transcript coords on the RC copy mirror within the exon: [20-8, 20-2) = [12, 18)
    assert r[0].split("\t")[3:5] == ["13", "18"] and r[0].split("\t")[6] == "-"


def test_rewrite_fasta_with_rc(tmp_path):
    fa = tmp_path / "in.fa"
    fa.write_text(">U1 extra\nACGTT\nAA\n>P1\nGGG\n")
    out = tmp_path / "out.fa"
    n = rewrite_fasta(fa, out, with_rc_copies({"U1"}))
    recs = dict(iter_fasta(out))
    assert n == 3
    assert recs["U1"] == "ACGTTAA" and recs["U1__rc"] == rev_comp("ACGTTAA") == "TTAACGT"
    assert recs["P1"] == "GGG"


# ---------- flanks ----------

# 40 bp: A[0,10) C[10,20) G[20,30) T[30,40)
GENOME = DictGenome({"c1": "AAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTTTTTTT"})


def test_build_flank_plus_strand():
    tx = _tx(strand="+", exons=((10, 20),))               # the C block
    fi = build_flank(tx, GENOME, k5=4, k3=5)
    assert fi.prefix == "AAAA" and fi.suffix == "GGGGG"
    assert fi.n_pad5 == 0 and fi.n_pad3 == 0
    assert fi.model.exons == [(6, 25)]
    assert fi.padded_record("C" * 10) == "AAAA" + "C" * 10 + "GGGGG"
    assert fi.total_len == 19


def test_build_flank_minus_strand_is_reverse_complemented():
    tx = _tx(strand="-", exons=((10, 20),))               # transcript seq = RC(C*10) = G*10
    fi = build_flank(tx, GENOME, k5=3, k3=4)
    # 5p flank in transcript orientation = RC of genomic [20,23) = RC("GGG") = "CCC"
    assert fi.prefix == "CCC"
    # 3p flank = RC of genomic [6,10) = RC("AAAA") = "TTTT"
    assert fi.suffix == "TTTT"
    assert fi.model.exons == [(6, 23)]


def test_build_flank_clamps_at_contig_edge_with_n_pad():
    tx = _tx(strand="+", exons=((2, 20),))
    fi = build_flank(tx, GENOME, k5=5, k3=30)
    assert fi.prefix == "AA" and fi.n_pad5 == 3
    assert fi.suffix == "G" * 10 + "T" * 10 and fi.n_pad3 == 10
    assert fi.model.exons == [(0, 40)]
    rec = fi.padded_record("X" * 18)
    assert rec.startswith("NNNAA") and rec.endswith("N" * 10)
    assert len(rec) == fi.total_len == 5 + 18 + 30


def test_build_flank_multi_exon_extends_only_terminal_exons():
    tx = _tx(strand="+", exons=((10, 14), (20, 24), (30, 34)))
    fi = build_flank(tx, GENOME, k5=2, k3=2)
    assert fi.model.exons == [(8, 14), (20, 24), (30, 36)]


def test_flank_orf_projects_into_genomic_flank():
    tx = _tx(strand="+", exons=((10, 20),))
    fi = build_flank(tx, GENOME, k5=4, k3=5)
    # ORF covering padded record [2, 17): starts in 5p flank (2 bases), ends in 3p flank (3 bases)
    pl = place_orf([(2, 17)], fi)
    assert (pl.ext5, pl.ext3, pl.tx_overlap, pl.coding_len) == (2, 3, 10, 15)
    assert pl.completion == "5p3p" and not pl.in_npad
    assert (pl.tx_start, pl.tx_end) == (-2, 13)
    lines = project_tx_intervals_to_genomic("T1", pl.model_intervals(fi), fi.model, "x")
    assert len(lines) == 1
    f = lines[0].split("\t")
    assert f[3:5] == ["9", "23"]        # genomic 0-based [8, 23) -> 1-based 9..23


def test_place_orf_inside_transcript_has_no_extension():
    tx = _tx(strand="+", exons=((10, 20),))
    fi = build_flank(tx, GENOME, k5=4, k3=5)
    pl = place_orf([(5, 11)], fi)
    assert pl.completion is None and pl.tx_start == 1 and pl.tx_end == 7
    assert pl.model_intervals(fi) == [(5, 11)]


def test_place_orf_in_n_pad_is_flagged():
    tx = _tx(strand="+", exons=((2, 20),))
    fi = build_flank(tx, GENOME, k5=5, k3=0)
    assert place_orf([(1, 12)], fi).in_npad          # starts in the 3 Ns
    assert not place_orf([(3, 12)], fi).in_npad
    fi_n = n_pad_only(tx, 5)
    assert place_orf([(2, 12)], fi_n).in_npad and not place_orf([(5, 12)], fi_n).in_npad


def test_clip_5p_matches_frame():
    # start 1 in the flank (k=4): keep from 4; 4 - 1 = 3 is a codon boundary
    assert clip_5p([(1, 20)], 4) == [(4, 20)]
    # start 2: next in-frame boundary at or after 4 is 5
    assert clip_5p([(2, 20)], 4) == [(5, 20)]
    # entirely inside the flank
    assert clip_5p([(0, 3)], 4) is None
    # first segment entirely in flank, second kept; frame from original start 0: boundary 6 ok
    assert clip_5p([(0, 3), (6, 12)], 4) == [(6, 12)]


def test_pick_orientation():
    assert pick_orientation(300, None) == "fwd"
    assert pick_orientation(None, 300) == "rc"
    assert pick_orientation(300, 300) == "fwd"
    assert pick_orientation(300, 301) == "rc"


# ---------- resolve_orf policy ----------

from drusilla.data.tx_augment import resolve_orf  # noqa: E402


def _fi():
    tx = _tx(strand="+", exons=((10, 20),))                # tx proper = padded [4, 14)
    return tx, build_flank(tx, GENOME, k5=4, k3=5)


def test_resolve_unpadded_passthrough():
    tx = _tx()
    r = resolve_orf([(2, 8)], tx, None)
    assert r.outcome == "unpadded" and r.intervals == [(2, 8)] and r.model is tx
    assert (r.tx_start, r.tx_end) == (2, 8) and r.completion is None


def test_resolve_inside_transcript_shifts_coords():
    tx, fi = _fi()
    r = resolve_orf([(5, 11)], tx, fi)
    assert r.outcome == "unchanged" and r.intervals == [(1, 7)] and r.model is tx
    assert (r.tx_start, r.tx_end) == (1, 7)


def test_resolve_5p_flank_policies():
    tx, fi = _fi()
    cds = [(2, 11)]                                         # starts 2 bases into the 5p flank
    assert resolve_orf(cds, tx, fi) == "dropped_in_flank"
    r = resolve_orf(cds, tx, fi, clip=True)
    assert r.outcome == "clipped_5p" and r.model is tx
    # clip at 4: (4-2)%3 -> shift 1 -> start 5 -> tx coord 1
    assert r.intervals == [(1, 7)] and r.completion is None
    r = resolve_orf(cds, tx, fi, keep=True)
    assert r.outcome == "completed" and r.completion == "5p" and r.model is fi.model
    assert r.intervals == [(2, 11)] and (r.tx_start, r.tx_end) == (-2, 7)


def test_resolve_3p_flank_policies():
    tx, fi = _fi()
    cds = [(6, 17)]                                         # ends 3 bases into the 3p flank
    assert resolve_orf(cds, tx, fi) == "dropped_3p_flank"
    assert resolve_orf(cds, tx, fi, clip=True) == "dropped_3p_flank"
    r = resolve_orf(cds, tx, fi, keep=True)
    assert r.completion == "3p" and r.intervals == [(6, 17)]


def test_resolve_keep_requires_transcript_support():
    tx, fi = _fi()
    # 2 of 10 coding bases inside the transcript: flank-only under the default 0.5
    assert resolve_orf([(0, 6)], tx, fi, keep=True) == "dropped_flank_only"
    r = resolve_orf([(0, 6)], tx, fi, keep=True, min_tx_frac=0.1)
    assert r.outcome == "completed"


def test_resolve_n_pad_is_dropped_before_anything_else():
    tx = _tx(strand="+", exons=((2, 20),))
    fi = build_flank(tx, GENOME, k5=5, k3=0)               # 3 Ns + "AA"
    assert resolve_orf([(1, 12)], tx, fi, keep=True) == "dropped_in_npad"
    fi_n = n_pad_only(tx, 5)
    assert resolve_orf([(3, 12)], tx, fi_n, keep=True) == "dropped_in_npad"
    assert resolve_orf([(5, 12)], tx, fi_n).outcome == "unchanged"

