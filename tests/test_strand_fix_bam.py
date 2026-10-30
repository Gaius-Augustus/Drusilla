"""Unit tests for the pure helpers of drusilla.data.strand_fix_bam (no pysam)."""

from __future__ import annotations

import re

from drusilla.data import strand_fix_bam as m


def test_read1_forward():
    # single-end: own orientation
    assert m.read1_forward(False, False, False) is True
    assert m.read1_forward(False, False, True) is False
    # paired read1: own orientation
    assert m.read1_forward(True, True, False) is True
    # paired read2: opposite of own orientation
    assert m.read1_forward(True, False, False) is False
    assert m.read1_forward(True, False, True) is True


def test_classify_run():
    assert m.classify_run(100, 95, 2000, 0.9) == m.UNKNOWN
    assert m.classify_run(5000, 4800, 2000, 0.9) == m.FR
    assert m.classify_run(5000, 100, 2000, 0.9) == m.RF
    assert m.classify_run(5000, 2500, 2000, 0.9) == m.UNSTRANDED
    assert m.classify_run(5000, 4000, 2000, 0.9) == m.UNSTRANDED   # 0.8 < 0.9


def test_transcript_strand():
    # dUTP (RF): read1 antisense -> read1 forward means transcript on '-'
    assert m.transcript_strand(True, m.RF) == "-"
    assert m.transcript_strand(False, m.RF) == "+"
    assert m.transcript_strand(True, m.FR) == "+"
    assert m.transcript_strand(False, m.FR) == "-"
    assert m.transcript_strand(True, m.UNSTRANDED) is None


def test_run_id():
    pat = re.compile(m.DEFAULT_RUN_REGEX)
    assert m.run_id("SRR31344714.1219032", pat) == "SRR31344714"
    assert m.run_id("ERR123.5/1", pat) == "ERR123"
    assert m.run_id("read_without_run", pat) == "_unmatched"


def test_write_report(tmp_path):
    counts = {"SRR1": [5000, 100], "SRR2": [10, 5]}
    calls = {"SRR1": m.RF, "SRR2": m.UNKNOWN}
    stats = {"SRR1": [6000, 5900, 0], "SRR2": [20, 0, 0]}
    out = tmp_path / "r.tsv"
    assert m.write_report(out, counts, calls, stats) == (6020, 5900, 0)
    lines = out.read_text().splitlines()
    assert lines[0].split("\t")[:4] == ["run", "n_spliced", "frac_read1_eq_xs", "call"]
    assert lines[1].split("\t") == ["SRR1", "5000", "0.020", "RF", "6000", "5900", "0"]
    assert lines[-1].startswith("#total\t")
