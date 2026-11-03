"""Strand '.' StringTie transcripts are labelled in both orientations (label policy v2):
the forward record as gffread extracts it, plus a reverse-complement record ``<tid>__rc``."""

from __future__ import annotations

from pathlib import Path

from drusilla.data.label_transcripts import (
    IR,
    START,
    STOP,
    project_labels,
)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_unstranded_transcript_is_labelled_as_forward(tmp_path):
    # gffread extracts a '.' transcript as the forward sequence, so a '+'
    # reference CDS inside its span must be labelled (not antisense_ir).
    st = tmp_path / "st.gtf"
    _write(st, ['chr1\tStringTie\texon\t1\t30\t.\t.\t.\ttranscript_id "ST.1";'])
    ref = tmp_path / "ref.gtf"
    _write(ref, ['chr1\tRef\tCDS\t4\t12\t.\t+\t0\ttranscript_id "R.1";'])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1, "antisense_ir": 1, "rc_copies": 1}
    lab = res.labels["ST.1"]
    assert lab[3] == START and lab[11] == STOP and (lab[:3] == IR).all()
    # the reverse-complement record only sees the gene antisense
    assert (res.labels["ST.1__rc"] == IR).all()
