"""Unit tests for label projection.

Conventions in fixtures:
* GTF is 1-based inclusive.
* Sequence labels: IR=0, START=1, E1=2, E2=3, E0=4, STOP=5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drusilla.data.label_transcripts import (
    E0,
    E1,
    E2,
    IR,
    START,
    STOP,
    build_labels,
    parse_reference_cds,
    parse_stringtie_gtf,
    project_labels,
    write_labelled_fasta,
)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- build_labels ----------

def test_build_labels_example_from_spec():
    # A 12 nt "transcript" with a 9 nt ORF starting at position 2:
    #   positions:   0 1 2     3 4 5 6 7 8 9 10   11
    #   labels   :   IR IR START E1 E2 E0 E1 E2 E0 E1 STOP IR
    labels = build_labels(12, orf=(2, 10))
    expected = np.array([
        IR, IR, START, E1, E2, E0, E1, E2, E0, E1, STOP, IR,
    ], dtype=np.int8)
    assert np.array_equal(labels, expected)


def test_build_labels_all_ir():
    labels = build_labels(5)
    assert np.array_equal(labels, np.zeros(5, dtype=np.int8))


# ---------- parsing ----------

def test_parse_stringtie_gtf_collects_exons(tmp_path):
    gtf = tmp_path / "st.gtf"
    _write(gtf, [
        'chr1\tStringTie\ttranscript\t10\t200\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t150\t200\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    st = parse_stringtie_gtf(gtf)
    assert list(st) == ["ST.1"]
    t = st["ST.1"]
    assert t.contig == "chr1"
    assert t.strand == "+"
    assert t.exons == [(9, 50), (149, 200)]
    assert t.length == (50 - 9) + (200 - 149)


def test_parse_reference_cds_merges_stop_codon(tmp_path):
    gff = tmp_path / "ref.gtf"
    _write(gff, [
        'chr1\tRef\tCDS\t100\t150\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tstop_codon\t151\t153\t.\t+\t0\ttranscript_id "R.1";',
    ])
    refs = parse_reference_cds(gff)
    r = refs["R.1"]
    assert r.cds_parts == [(99, 153)]
    assert r.has_stop_codon is True
    assert r.total_cds_len == 54
    assert r.complete  # 54 % 3 == 0, length >= 6


# ---------- end-to-end projection ----------

@pytest.fixture
def simple_plus_fixture(tmp_path):
    """A 100 nt single-exon StringTie transcript on chr1 '+'; a contained
    reference CDS of length 9 (1 ATG + 1 codon + 1 stop) at positions 50..58
    in genomic 1-based inclusive, which is 49..57 in 0-based inclusive, which
    is transcript coords 39..47 (exon starts at 10).
    """
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t109\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # 9 nt CDS including stop: ATG + 1 codon + stop => 9 bases total
        'chr1\tRef\tCDS\t50\t58\t.\t+\t0\ttranscript_id "R.1";',
    ])
    return st, ref


def test_simple_plus_strand_contained(simple_plus_fixture):
    st, ref = simple_plus_fixture
    res = project_labels(st, ref)

    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    assert labels.shape == (100,)

    # Exon starts at g=9 (0-based). CDS at g=49..57 -> t=40..48.
    orf_s, orf_e = 40, 48
    assert labels[orf_s] == START
    assert labels[orf_e] == STOP
    # Frame cycle between them: E1 E2 E0 E1 E2 E0 E1
    assert list(labels[orf_s + 1: orf_e]) == [E1, E2, E0, E1, E2, E0, E1]
    # Everything else is IR.
    assert (labels[:orf_s] == IR).all()
    assert (labels[orf_e + 1:] == IR).all()
    assert res.chosen_ref["ST.1"] == "R.1"


def test_no_overlap_is_ir_only(tmp_path):
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t1000\t1008\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"ir_only": 1}
    assert (res.labels["ST.1"] == IR).all()


def test_antisense_only_is_ir(tmp_path):
    # Only an opposite-strand ref overlaps -> emit all-IR labels (antisense
    # rescue). The transcript is kept in training, not dropped.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t28\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"antisense_ir": 1}
    assert "ST.1" in res.labels
    assert (res.labels["ST.1"] == IR).all()
    # No ref was chosen and the transcript is not flagged as partial.
    assert "ST.1" not in res.chosen_ref
    assert "ST.1" not in res.partial


def test_partial_overlap_3prime_truncation_is_kept(tmp_path):
    # StringTie exon g 10..50 -> 0-based [9, 50), length 41, t=0..40.
    # Ref CDS g 40..60 on '+' -> 0-based [39, 60), 21 nt long (complete).
    # Visible portion on transcript: g 39..49 -> t = 30..40 (length 11,
    # ref positions 0..10). 3' end of ref CDS is NOT visible.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t40\t60\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    assert labels.shape == (41,)
    # Visible region starts at ref pos 0 -> START at t=30, then frame cycle.
    assert labels[30] == START
    assert list(labels[31:41]) == [E1, E2, E0, E1, E2, E0, E1, E2, E0, E1]
    # No STOP because the 3' end of the ref CDS is not visible.
    assert STOP not in labels
    # Flanking region is IR.
    assert (labels[:30] == IR).all()

    (p,) = res.partial["ST.1"]
    assert p["has_5p"] is True
    assert p["has_3p"] is False
    assert (p["t_start"], p["t_end"]) == (30, 40)
    assert (p["ref_start_pos"], p["ref_end_pos"]) == (0, 10)
    assert p["ref_total_len"] == 21
    assert res.chosen_ref["ST.1"] == "R.1"


def test_partial_overlap_5prime_truncation_is_kept(tmp_path):
    # StringTie exon g 30..100 -> 0-based [29, 100), length 71, t=0..70.
    # Ref CDS g 10..39 on '+' -> 0-based [9, 39), 30 nt long (complete).
    # Visible portion: g 29..38 -> ref positions 20..29 (5' is cut off),
    # t = 0..9. 3' end of ref CDS IS visible -> STOP at t=9.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t30\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t10\t39\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    # ref pos 29 (the last base) -> STOP. ref pos 20..28 use the frame cycle.
    assert labels[9] == STOP
    # frame at t=k corresponds to ref_pos = 20 + k for k in 0..8.
    # _FRAME_CYCLE[(rp-1) % 3]: rp=20 -> FC[1]=E2, rp=21 -> FC[2]=E0, rp=22 -> FC[0]=E1, ...
    assert list(labels[0:9]) == [E2, E0, E1, E2, E0, E1, E2, E0, E1]
    assert (labels[10:] == IR).all()
    # No START because ref pos 0 is not visible.
    assert START not in labels

    (p,) = res.partial["ST.1"]
    assert p["has_5p"] is False
    assert p["has_3p"] is True
    assert (p["ref_start_pos"], p["ref_end_pos"]) == (20, 29)
    assert p["ref_total_len"] == 30


def test_refseq_partial_or_pseudo_cds_is_not_complete(tmp_path):
    # RefSeq GFF3: a CDS flagged partial=true (no verified start) or
    # pseudo=true is incomplete even when its length is a multiple of 3, so
    # the transcript gets all-IR labels instead of a START at a wrong base.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t200\t.\t+\t.\ttranscript_id "ST.1";',
        'chr2\tStringTie\texon\t1\t200\t.\t+\t.\ttranscript_id "ST.2";',
    ])
    ref = tmp_path / "ref.gff"
    _write(ref, [
        "chr1\tRef\tCDS\t10\t18\t.\t+\t0\tID=cds-1;Parent=R.1;partial=true;start_range=.,10",
        "chr2\tRef\tCDS\t10\t18\t.\t+\t0\tID=cds-2;Parent=R.2;pseudo=true",
    ])
    refs = parse_reference_cds(ref)
    assert refs["R.1"].flagged_incomplete and not refs["R.1"].complete
    assert refs["R.2"].flagged_incomplete and not refs["R.2"].complete
    res = project_labels(st, ref)
    assert res.stats == {"ref_partial_ir": 2}
    assert (res.labels["ST.1"] == IR).all() and (res.labels["ST.2"] == IR).all()


def test_only_partial_refs_is_ir(tmp_path):
    # CDS length 10 (not divisible by 3) -> ref.complete == False.
    # No complete same-strand ref overlaps -> emit all-IR labels (the reading
    # frame is not derivable from a partial ref, so we don't fabricate one).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t29\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"ref_partial_ir": 1}
    assert (res.labels["ST.1"] == IR).all()
    assert "ST.1" not in res.partial


def test_two_separate_genes_are_both_labelled_as_fused(tmp_path):
    # Two contained ref CDSes that do not overlap on the transcript:
    # R.short (9 nt) and R.long (15 nt). Both are labelled, in transcript order.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1	StringTie	exon	1	200	.	+	.	transcript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1	Ref	CDS	10	18	.	+	0	transcript_id "R.short";',
        'chr1	Ref	CDS	100	114	.	+	0	transcript_id "R.long";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_fused": 1}
    assert res.chosen_ref["ST.1"] == "R.short,R.long"
    labels = res.labels["ST.1"]
    assert labels[9] == START and labels[17] == STOP
    assert labels[99] == START and labels[113] == STOP
    assert (labels[18:99] == IR).all()


def test_overlapping_isoforms_keep_only_the_longest(tmp_path):
    # Two contained CDSes of one gene (alternative start): they overlap on
    # the transcript, so only the longest is labelled.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1	StringTie	exon	1	200	.	+	.	transcript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1	Ref	CDS	13	30	.	+	0	transcript_id "R.short";',
        'chr1	Ref	CDS	10	30	.	+	0	transcript_id "R.long";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_multi": 1}
    assert res.chosen_ref["ST.1"] == "R.long"
    labels = res.labels["ST.1"]
    assert labels[9] == START and labels[29] == STOP
    assert (labels[:9] == IR).all() and (labels[30:] == IR).all()


def test_fused_transcript_gets_truncated_second_gene_as_partial(tmp_path):
    # A complete gene R.1 and a second gene R.2 truncated by the transcript's
    # 3' end: R.2's visible run is labelled too (no START, no STOP), while a
    # truncated isoform R.1b overlapping R.1 is skipped.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1	StringTie	exon	1	100	.	+	.	transcript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1	Ref	CDS	10	18	.	+	0	transcript_id "R.1";',
        'chr1	Ref	CDS	10	120	.	+	0	transcript_id "R.1b";',
        'chr1	Ref	CDS	80	130	.	+	0	transcript_id "R.2";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1, "extra_partial": 1}
    assert res.chosen_ref["ST.1"] == "R.1,R.2"
    labels = res.labels["ST.1"]
    assert labels[9] == START and labels[17] == STOP
    # R.2: g 80..130 (0-based [79, 130)), visible g 79..99 -> t 79..99.
    assert labels[79] == START
    assert (labels[80:100] != IR).all() and STOP not in labels[79:100]
    assert (labels[18:79] == IR).all()
    (p,) = res.partial["ST.1"]
    assert p["ref_tid"] == "R.2" and p["has_5p"] is True and p["has_3p"] is False


def test_minus_strand_contained(tmp_path):
    # Single-exon minus-strand StringTie transcript on chr1, genomic 1..30.
    # 0-based: exon [0, 30), transcript length 30.  strand '-' => transcript
    # coordinate 0 corresponds to genomic position 29.
    # Reference CDS at genomic 10..18 (1-based incl), i.e. 0-based [9, 18).
    # On the transcript, the CDS maps to t-positions 29-9=20 down to 29-17=12,
    # sorted ascending that's 12..20 inclusive -> orf_start=12, orf_end=20.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t30\t.\t-\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t10\t18\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    assert labels[12] == START
    assert labels[20] == STOP
    assert list(labels[13:20]) == [E1, E2, E0, E1, E2, E0, E1]


def test_plus_strand_cds_spanning_matching_intron(tmp_path):
    # StringTie has two exons: [0, 20) and [30, 50) (genomic 0-based).
    # Reference CDS has two parts: [10, 20) and [30, 40).  Their introns
    # match, so in transcript coords the two CDS parts are contiguous
    # (t: 10..19 and 20..29 -> ORF from t=10 to t=29, length 20... but
    # 20 is not divisible by 3, so for this test we shrink to 18).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t20\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t31\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # CDS parts: g [10, 20) (len 10) and g [30, 38) (len 8) -> total 18.
        # Ref intron [20, 30) == StringTie intron [20, 30), so contiguous in t.
        'chr1\tRef\tCDS\t11\t20\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tCDS\t31\t38\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    # Transcript length = 20 + 20 = 40.  First-exon CDS starts at t=10
    # (g=10 lies at exon-1 offset 10).  ORF length 18 -> ends at t=27.
    assert labels[10] == START
    assert labels[27] == STOP
    # Check contiguity: all positions between must be coding, not IR.
    assert (labels[10:28] != IR).all()


def test_plus_strand_cds_spanning_nonmatching_intron_is_kept_partial(tmp_path):
    # StringTie single exon covers everything [0, 50).  Reference CDS has
    # two parts with a genomic intron (20..29) in between that StringTie
    # treats as part of the exon -> CDS is non-contiguous in transcript
    # coords. The partial rescue labels the longest contiguous run.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # Parts g [10, 19) + g [29, 38); total 18 (%3==0, complete).
        'chr1\tRef\tCDS\t11\t19\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tCDS\t30\t38\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    # Two equal-length runs (t=10..18 ref pos 0..8; t=29..37 ref pos 9..17).
    # Tie-broken to the first (5'-anchored) run -> START at t=10.
    assert labels[10] == START
    assert list(labels[11:18]) == [E1, E2, E0, E1, E2, E0, E1]
    # ref pos 8 != ref_total-1 -> not STOP, just frame.
    assert labels[18] != STOP
    assert labels[18] != IR
    # The second half (real CDS in genomic terms) gets IR in transcript coords
    # because StringTie's missing splice breaks contiguity.
    assert (labels[19:] == IR).all()

    (p,) = res.partial["ST.1"]
    assert p["has_5p"] is True
    assert p["has_3p"] is False


def test_minus_strand_partial_overlap_is_kept(tmp_path):
    # Minus-strand single exon g 1..30 -> 0-based [0, 30), t length 30.
    # Ref CDS on '-' strand at g 20..40 -> 0-based [19, 40), 21 nt (complete).
    # Visible portion: g 19..29 -> ref positions 10..20 (5' of ref cut off),
    # corresponding to t positions 0..10 (since t=29-g for single exon).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t30\t.\t-\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t40\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    (p,) = res.partial["ST.1"]
    # 3' end of ref (last base) is at lowest genomic position; for a -strand
    # ref at g [19, 40) the last base is g=19 -> visible on this transcript
    # -> has_3p should be True; the 5' end (g=39) is not visible -> has_5p
    # should be False.
    assert p["has_5p"] is False
    assert p["has_3p"] is True
    assert p["ref_total_len"] == 21
    # The visible region must contain a STOP label at the 3' end.
    labels = res.labels["ST.1"]
    assert (labels == STOP).any()
    assert START not in labels


def test_unstranded_transcript_is_labelled_in_both_orientations(tmp_path):
    # StringTie "." strand: gffread extracts the forward sequence, so a "+"
    # ref contained in the span is labelled on the forward copy; the
    # reverse-complement copy ST.1__rc only sees an antisense gene.
    st = tmp_path / "st.gtf"
    _write(st, [
        "chr1	StringTie	exon	1	30	.	.	.	transcript_id \"ST.1\";",
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        "chr1	Ref	CDS	4	12	.	+	0	transcript_id \"R.1\";",
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1, "antisense_ir": 1, "rc_copies": 1}
    lab = res.labels["ST.1"]
    assert lab[3] == START and lab[11] == STOP and (lab[:3] == IR).all()
    assert (res.labels["ST.1__rc"] == IR).all()
    assert "ST.1__rc" not in res.chosen_ref


def test_unstranded_transcript_minus_gene_lands_on_rc_copy(tmp_path):
    # Same transcript, gene on '-': the forward copy is antisense, the
    # reverse-complement copy carries the ORF. Transcript g [0, 30) read as
    # '-': t = 29 - g, so CDS 0-based [3, 12) maps to t 18..26.
    st = tmp_path / "st.gtf"
    _write(st, [
        "chr1	StringTie	exon	1	30	.	.	.	transcript_id \"ST.1\";",
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        "chr1	Ref	CDS	4	12	.	-	0	transcript_id \"R.1\";",
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1, "antisense_ir": 1, "rc_copies": 1}
    assert (res.labels["ST.1"] == IR).all()
    rc = res.labels["ST.1__rc"]
    assert rc[18] == START and rc[26] == STOP
    assert list(rc[19:26]) == [E1, E2, E0, E1, E2, E0, E1]
    assert res.chosen_ref["ST.1__rc"] == "R.1"

    single = project_labels(st, ref, both_orientations=False)
    assert single.stats == {"antisense_ir": 1}
    assert set(single.labels) == {"ST.1"}


def test_write_labelled_fasta_adds_reverse_complement_from_gzip(tmp_path):
    import gzip

    fa = tmp_path / "tx.fa.gz"
    with gzip.open(fa, "wt") as fh:
        fh.write(">ST.1 extra words\nACGTTT\nGGA\n>ST.2\nCCC\n>ST.3\nAAA\n")
    out = tmp_path / "labelled.fa"
    n = write_labelled_fasta(fa, out, {"ST.1", "ST.1__rc", "ST.3__rc"}, width=4)
    assert n == 3
    assert out.read_text() == (
        ">ST.1\nACGT\nTTGG\nA\n"
        ">ST.1__rc\nTCCA\nAACG\nT\n"
        ">ST.3__rc\nTTT\n"
    )
