"""Recover library strandedness in a pooled RNA-seq BAM.

Aligners run without a strandedness option (e.g. HISAT2 without
``--rna-strandness``, as in VARUS) tag only spliced reads with an ``XS:A``
strand, inferred from the splice motif. StringTie then emits strand ``.``
for almost every single-exon transcript and cannot separate overlapping
antisense genes, even when the underlying runs were stranded.

This module infers the protocol of every sequencing run from its spliced
reads (does the first read of a fragment map to the XS strand or the
opposite one?) and then sets ``XS:A`` on *all* mapped reads of stranded
runs, unspliced ones included. StringTie in its default (unstranded) mode
uses the tag of every read, so single-exon transcripts get a strand and
antisense neighbours separate. Reads of unstranded runs are left untouched
(or dropped with ``drop_unstranded=True``).

Run ids are taken from the read name (``SRR123.456`` -> ``SRR123`` with the
default regex); pass another ``run_regex`` for other naming schemes, or
``"none"`` to treat the whole BAM as one run (a BAM of a single library).

CLI: ``drusilla strand-fix-bam``; also ``drusilla annotate --bam ... --fix-strand``.
Requires ``pysam`` (``pip install "drusilla[bam]"``).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

FR = "FR"            # read1 maps to the transcript strand (ligation / "secondstrand")
RF = "RF"            # read1 maps antisense to the transcript (dUTP, "firststrand"; common)
UNSTRANDED = "unstranded"
UNKNOWN = "unknown"  # too few spliced reads to decide

DEFAULT_RUN_REGEX = r"^([A-Za-z]+\d+)"
DEFAULT_MIN_SPLICED = 2000
DEFAULT_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# pure helpers (unit-tested, no pysam)
# ---------------------------------------------------------------------------

def read1_forward(is_paired: bool, is_read1: bool, is_reverse: bool) -> bool:
    """Orientation of the fragment's first read: True if read1 maps forward.
    For read2 the first read maps the opposite way."""
    if not is_paired or is_read1:
        return not is_reverse
    return is_reverse


def classify_run(n_spliced: int, n_agree: int, min_spliced: int, threshold: float) -> str:
    """Protocol call from spliced-read agreement between read1 orientation and XS."""
    if n_spliced < min_spliced:
        return UNKNOWN
    frac = n_agree / n_spliced
    if frac >= threshold:
        return FR
    if frac <= 1.0 - threshold:
        return RF
    return UNSTRANDED


def transcript_strand(r1_forward: bool, call: str) -> str | None:
    """XS value implied by the fragment orientation under a protocol call."""
    if call == FR:
        return "+" if r1_forward else "-"
    if call == RF:
        return "-" if r1_forward else "+"
    return None


def run_id(qname: str, pattern: re.Pattern | None) -> str:
    """Run id of a read; ``pattern=None`` puts every read into the one run ``all``."""
    if pattern is None:
        return "all"
    m = pattern.match(qname)
    return m.group(1) if m else "_unmatched"


# ---------------------------------------------------------------------------
# BAM passes (pysam)
# ---------------------------------------------------------------------------

def _usable(read) -> bool:
    return not (read.is_unmapped or read.is_secondary or read.is_supplementary
                or read.is_duplicate or read.is_qcfail)


def pass1_counts(in_bam: str, pattern: re.Pattern, max_reads: int | None,
                 threads: int) -> dict[str, list[int]]:
    """Per run: [n_spliced_with_XS, n_read1_orientation_equals_XS]."""
    import pysam
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    n = 0
    with pysam.AlignmentFile(in_bam, "rb", threads=threads) as fh:
        for read in fh.fetch(until_eof=True):
            n += 1
            if max_reads is not None and n > max_reads:
                break
            if not _usable(read):
                continue
            if "N" not in read.cigarstring:
                continue
            try:
                xs = read.get_tag("XS")
            except KeyError:
                continue
            if xs not in ("+", "-"):
                continue
            c = counts[run_id(read.query_name, pattern)]
            c[0] += 1
            r1f = read1_forward(read.is_paired, read.is_read1, read.is_reverse)
            if (xs == "+") == r1f:
                c[1] += 1
    return counts


def pass2_write(in_bam: str, out_bam: str, calls: dict[str, str], pattern: re.Pattern,
                drop_unstranded: bool, threads: int) -> dict[str, list[int]]:
    """Rewrite the BAM; returns per-run [n_reads, n_tagged, n_dropped]."""
    import pysam
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    with pysam.AlignmentFile(in_bam, "rb", threads=threads) as fin, \
            pysam.AlignmentFile(out_bam, "wb", template=fin, threads=threads) as fout:
        for read in fin.fetch(until_eof=True):
            rid = run_id(read.query_name, pattern)
            call = calls.get(rid, UNKNOWN)
            st = stats[rid]
            st[0] += 1
            if call in (FR, RF):
                if not read.is_unmapped:
                    r1f = read1_forward(read.is_paired, read.is_read1, read.is_reverse)
                    read.set_tag("XS", transcript_strand(r1f, call), value_type="A")
                    st[1] += 1
            elif drop_unstranded:
                st[2] += 1
                continue
            fout.write(read)
    return stats


def write_report(report_tsv: Path, counts: dict[str, list[int]], calls: dict[str, str],
                 stats: dict[str, list[int]]) -> tuple[int, int, int]:
    """Write the per-run TSV; returns (total_reads, tagged_reads, dropped_reads)."""
    total = sum(s[0] for s in stats.values())
    tagged = sum(s[1] for s in stats.values())
    dropped = sum(s[2] for s in stats.values())
    with open(report_tsv, "w") as fh:
        fh.write("run\tn_spliced\tfrac_read1_eq_xs\tcall\tn_reads\tn_tagged\tn_dropped\n")
        for rid in sorted(stats, key=lambda r: -stats[r][0]):
            n, a = counts.get(rid, [0, 0])
            frac = f"{a / n:.3f}" if n else "NA"
            s = stats[rid]
            fh.write(f"{rid}\t{n}\t{frac}\t{calls.get(rid, UNKNOWN)}\t{s[0]}\t{s[1]}\t{s[2]}\n")
        fh.write(f"#total\t\t\t\t{total}\t{tagged}\t{dropped}\n")
    return total, tagged, dropped


def fix_strand(
    in_bam: str | Path,
    out_bam: str | Path,
    report_tsv: str | Path,
    *,
    run_regex: str = DEFAULT_RUN_REGEX,
    min_spliced: int = DEFAULT_MIN_SPLICED,
    threshold: float = DEFAULT_THRESHOLD,
    max_pass1_reads: int | None = None,
    drop_unstranded: bool = False,
    threads: int = 4,
    index: bool = True,
) -> dict[str, int]:
    """Two-pass strand recovery. Returns summary counts.

    ``run_regex``: regex whose group 1 is the run id in a read name; ``""`` or
    ``"none"`` makes the whole BAM one run.

    Keys: ``runs``, ``stranded_runs``, ``unstranded_runs``, ``unknown_runs``,
    ``reads``, ``tagged``, ``dropped``.
    """
    try:
        import pysam  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "strand fixing needs pysam: pip install pysam  (or drusilla[bam])"
        ) from e

    pattern = None if run_regex in ("", "none") else re.compile(run_regex)
    print(f"[strand-fix] pass 1: spliced-read orientation vs XS in {in_bam}", flush=True)
    counts = pass1_counts(str(in_bam), pattern, max_pass1_reads, threads)
    calls = {rid: classify_run(n, a, min_spliced, threshold)
             for rid, (n, a) in counts.items()}
    n_str = sum(1 for c in calls.values() if c in (FR, RF))
    n_uns = sum(1 for c in calls.values() if c == UNSTRANDED)
    n_unk = sum(1 for c in calls.values() if c == UNKNOWN)
    print(f"[strand-fix] runs seen={len(counts)} stranded={n_str} "
          f"unstranded={n_uns} unknown={n_unk}", flush=True)

    print(f"[strand-fix] pass 2: writing {out_bam}", flush=True)
    stats = pass2_write(str(in_bam), str(out_bam), calls, pattern, drop_unstranded, threads)
    total, tagged, dropped = write_report(Path(report_tsv), counts, calls, stats)
    print(f"[strand-fix] reads={total} tagged_from_stranded_runs={tagged} "
          f"({100 * tagged / max(1, total):.1f}%) dropped={dropped} -> {report_tsv}",
          flush=True)

    if index:
        import pysam
        try:
            pysam.index(str(out_bam))
        except Exception as e:  # noqa: BLE001 - the index is optional for StringTie
            print(f"[strand-fix] warn: indexing failed: {e}", file=sys.stderr)
    return {"runs": len(counts), "stranded_runs": n_str, "unstranded_runs": n_uns,
            "unknown_runs": n_unk, "reads": total, "tagged": tagged, "dropped": dropped}
