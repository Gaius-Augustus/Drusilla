"""`drusilla filter-subseq`: drop predicted isoforms whose CDS is a
(near-)sub-sequence of another isoform in the same locus.

Works on any prediction GTF with StringTie-style ``STRG.<gene>.<iso>``
transcript_ids (locus = transcript_id minus the trailing ``.<iso>``).
See ``drusilla.postprocess.subseq_filter`` for the matching rules and
tolerance semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..postprocess.subseq_filter import run_filter


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--orfs-gtf", type=Path, required=True,
                    help="Input prediction GTF (CDS lines).")
    ap.add_argument("--out-gtf", type=Path, required=True,
                    help="Filtered GTF (dropped isoforms removed).")
    ap.add_argument("--report-tsv", type=Path, default=None,
                    help="Optional TSV of dropped_tid \\t keeper_tid \\t reason.")
    ap.add_argument("--terminal-overhang-nt", type=int, default=0,
                    help="Overhang tolerance at A's leftmost/rightmost CDS "
                         "block past the matched B block (default 0 = strict).")
    ap.add_argument("--splice-shift-nt", type=int, default=0,
                    help="Tolerance at each side of an interior CDS block "
                         "boundary. Must be a multiple of 3 (default 0 = strict).")
    ap.add_argument("--allow-exon-skip", action="store_true",
                    help="Allow A's exons to be a non-contiguous ordered subset "
                         "of B's exons (default off).")


def run(args: argparse.Namespace) -> int:
    stats = run_filter(
        orfs_gtf=args.orfs_gtf,
        out_gtf=args.out_gtf,
        report_tsv=args.report_tsv,
        terminal_overhang_nt=args.terminal_overhang_nt,
        splice_shift_nt=args.splice_shift_nt,
        allow_exon_skip=args.allow_exon_skip,
    )
    print(f"[filter-subseq] tolerances     : "
          f"overhang={args.terminal_overhang_nt}nt "
          f"splice_shift={args.splice_shift_nt}nt "
          f"allow_exon_skip={args.allow_exon_skip}")
    print(f"[filter-subseq] input tx       : {stats['n_input_tx']}")
    print(f"[filter-subseq] dropped tx     : {stats['n_dropped_tx']} "
          f"({stats['pct_dropped']:.1f}%)")
    print(f"[filter-subseq] kept tx        : {stats['n_kept_tx']}")
    print(f"[filter-subseq] CDS kept/drop  : {stats['n_kept_cds']} / {stats['n_dropped_cds']}")
    print(f"[filter-subseq] out gtf        : {args.out_gtf}")
    if args.report_tsv is not None:
        print(f"[filter-subseq] report tsv     : {args.report_tsv}")
    return 0
