"""`drusilla strand-fix-bam`: set XS strand tags on all reads of stranded runs.

Use before StringTie when a BAM pools several sequencing runs that were
aligned without a strandedness option (VARUS / HISAT2 default). The
protocol of every run is inferred from its spliced reads; see
``drusilla.data.strand_fix_bam``.

    drusilla strand-fix-bam --in-bam aligned.bam --out-bam aligned.sfix.bam \\
        --report-tsv strand_runs.tsv
    stringtie aligned.sfix.bam -o stringtie.gtf

The same step runs inside ``drusilla annotate --bam ... --fix-strand``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data import strand_fix_bam as sfb


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--in-bam", required=True, type=Path,
                    help="Input BAM (any order; read once per pass).")
    ap.add_argument("--out-bam", required=True, type=Path,
                    help="Output BAM with XS:A set on all mapped reads of stranded runs.")
    ap.add_argument("--report-tsv", required=True, type=Path,
                    help="Per-run TSV: spliced reads, orientation fraction, call, "
                         "reads written/tagged/dropped.")
    ap.add_argument("--run-regex", default=sfb.DEFAULT_RUN_REGEX,
                    help="Regex whose group 1 is the run id in the read name "
                         "(default: leading letters+digits, e.g. SRR123456).")
    ap.add_argument("--min-spliced", type=int, default=sfb.DEFAULT_MIN_SPLICED,
                    help=f"Spliced reads needed to call a run (default {sfb.DEFAULT_MIN_SPLICED}).")
    ap.add_argument("--stranded-threshold", type=float, default=sfb.DEFAULT_THRESHOLD,
                    help="Agreement fraction for a stranded call; runs between "
                         f"1-t and t are unstranded (default {sfb.DEFAULT_THRESHOLD}).")
    ap.add_argument("--max-pass1-reads", type=int, default=None,
                    help="Stop pass 1 after this many records (speed; default all).")
    ap.add_argument("--drop-unstranded-runs", action="store_true",
                    help="Do not write reads of unstranded/unknown runs.")
    ap.add_argument("--threads", type=int, default=4,
                    help="BAM (de)compression threads (default 4).")
    ap.add_argument("--no-index", action="store_true",
                    help="Skip indexing the output BAM.")


def run(args: argparse.Namespace) -> int:
    sfb.fix_strand(
        args.in_bam, args.out_bam, args.report_tsv,
        run_regex=args.run_regex,
        min_spliced=args.min_spliced,
        threshold=args.stranded_threshold,
        max_pass1_reads=args.max_pass1_reads,
        drop_unstranded=args.drop_unstranded_runs,
        threads=args.threads,
        index=not args.no_index,
    )
    return 0
