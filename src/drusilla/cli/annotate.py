"""`drusilla annotate`: end-to-end ORF annotation from a StringTie GTF or BAM.

Two input modes:
    --stringtie-gtf <gtf> --genome <fa>      StringTie has already been run
    --bam <bam>          --genome <fa>       run StringTie internally first

Model weights are resolved via the registry: pass ``--model <name>`` to
download and use a released model, or ``--weights <path>`` to load a
locally-trained checkpoint.

Default output (in --out-dir):
    orfs.gtf        PRIMARY. Genomic-coordinate GTF of predicted CDS lines
                    (source = "drusilla"), 1-based inclusive, with the
                    subsequence-isoform collapse applied.
    orfs_local.gtf  Same predictions in transcript (local) coordinates:
                    one CDS line per predicted ORF, contig column holds
                    the StringTie transcript_id.
    orfs.log        b2m annotation log.

Subsequence collapse is ON by default. Disable with --no-subseq-collapse
(both orfs.gtf and orfs_local.gtf are then the raw predictions).
Opt into a drop-report TSV with --subseq-report.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def add_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stringtie-gtf", type=Path,
                     help="StringTie GTF (skip running StringTie).")
    src.add_argument("--bam", type=Path,
                     help="Sorted alignment BAM (run StringTie internally).")
    ap.add_argument("--transcripts-fa", type=Path, default=None,
                    help="FASTA of assembled transcripts (skips gffread).")
    ap.add_argument("--genome", type=Path, required=True,
                    help="Genome FASTA matching the GTF/BAM contig names.")

    # Model selection: either a registered name (auto-download) or a
    # local weights file (advanced / custom-trained).
    mdl = ap.add_mutually_exclusive_group()
    mdl.add_argument("--model", type=str, default=None,
                     help="Registered model name (see `drusilla models list`). "
                          "Defaults to 'vertebrates' if neither --model nor "
                          "--weights is given.")
    mdl.add_argument("--weights", type=Path, default=None,
                     help="Path to a local .weights.h5 file (advanced use).")
    ap.add_argument("--config", type=Path, default=None,
                    help="Architecture config YAML. Auto-selected from the "
                         "registry when --model is used; required with --weights.")

    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for orfs.gtf and intermediates.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Inference batch size. If omitted, auto-selected "
                         "from the detected GPU memory and the model's "
                         "chunk_len (see drusilla.gpu.compute_auto_batch_size).")
    ap.add_argument("--parallel", type=int, default=100,
                    help="HMM parallel scan factor (default 100).")
    ap.add_argument("--no-codon-emitter", action="store_true",
                    help="Disable hard ATG/STOP/in-frame-stop codon "
                         "constraints in the HMM (debugging variant).")
    ap.add_argument("--restrict-start-to-ir-start", action="store_true",
                    help="Restrict the HMM's initial-state distribution "
                         "to {IR, START}.")
    ap.add_argument("--ir-prior", type=float, default=0.5,
                    help="P(start in IR) under --restrict-start-to-ir-start.")
    ap.add_argument("--prefix-pad-n", type=int, default=0,
                    help="Prepend this many literal 'N' bases as left-context.")
    ap.add_argument("--flank-bp", type=int, default=0,
                    help="Prepend this many bp of real upstream genomic "
                         "context (RC for '-' strand).")
    ap.add_argument("--flank-clip", action="store_true",
                    help="With --flank-bp: clip ORFs whose START is in the "
                         "flank to the first in-frame codon inside the tx.")
    ap.add_argument("--threads", type=int, default=4,
                    help="Threads for stringtie/gffread.")
    ap.add_argument("--longread", action="store_true",
                    help="Pass -L to StringTie (long-read assembly mode).")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Do not delete intermediate stringtie.gtf / "
                         "transcripts.fa in --out-dir.")

    # Reference-free StringTie pre-filters.
    ap.add_argument("--min-cov", type=float, default=None,
                    help="Drop StringTie transcripts with cov < X.")
    ap.add_argument("--min-tpm", type=float, default=None,
                    help="Drop StringTie transcripts with TPM < X.")
    ap.add_argument("--drop-unstranded", action="store_true",
                    help="Drop StringTie transcripts with strand '.'.")
    ap.add_argument("--drop-single-exon", action="store_true",
                    help="Drop single-exon StringTie transcripts.")

    # Post-decode filters.
    ap.add_argument("--min-coding-length", type=int, default=200,
                    help="Drop predicted ORFs with total CDS length < X nt "
                         "(default 200). Set to 0 to disable.")
    ap.add_argument("--single-isoform", action="store_true",
                    help="Keep at most one transcript per StringTie gene_id "
                         "(longest predicted CDS wins).")
    ap.add_argument("--min-utr-5", type=int, default=0,
                    help="Drop ORFs whose START is within N nt of the tx 5' end.")
    ap.add_argument("--min-utr-3", type=int, default=0,
                    help="Drop ORFs whose STOP is within N nt of the tx 3' end.")
    ap.add_argument("--partial-out", type=Path, default=None,
                    help="Also write 3'-truncated ORFs to this GTF.")
    ap.add_argument("--partial5-out", type=Path, default=None,
                    help="Also write 5'-truncated ORFs to this GTF.")
    ap.add_argument("--lorf-class", action="store_true",
                    help="Annotate each ORF with its LORF class (LORF_UPSTOP / "
                         "sORF_UPSTOP / upLORF / LORF_NOUPSTOP) as a GTF attr.")

    # Subsequence-isoform collapse (ON by default).
    ap.add_argument("--no-subseq-collapse",
                    dest="subseq_collapse", action="store_false", default=True,
                    help="Disable the default subsequence-isoform collapse. "
                         "orfs.gtf / orfs_local.gtf are then the raw model "
                         "predictions (all isoforms kept).")
    ap.add_argument("--subseq-terminal-overhang-nt", type=int, default=0,
                    help="Subseq collapse: overhang tolerance at "
                         "leftmost/rightmost CDS block (default 0 = strict).")
    ap.add_argument("--subseq-splice-shift-nt", type=int, default=0,
                    help="Subseq collapse: interior boundary tolerance "
                         "(must be a multiple of 3; default 0 = strict).")
    ap.add_argument("--subseq-allow-exon-skip", action="store_true",
                    help="Subseq collapse: allow A's exons to be a "
                         "non-contiguous ordered subset of B's exons.")
    ap.add_argument("--subseq-report", action="store_true",
                    help="Also write orfs.dropped_subseq.tsv "
                         "(dropped_tid, keeper_tid, reason).")


def _enable_gpu_memory_growth() -> None:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _run_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True)


def _gtf_attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rev_comp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _load_tx_seqs(fa_path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def _build_flank_prefixes(
    stringtie_gtf: Path,
    genome_fa: Path,
    k: int,
) -> dict[str, str]:
    """Return ``{tx_id: K-bp upstream genomic prefix}`` for every tx."""
    from pyfaidx import Fasta

    tx_extent: dict[str, tuple[str, str, int, int]] = {}
    for raw in Path(stringtie_gtf).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        tid = _gtf_attr(f[8], "transcript_id")
        if tid is None:
            continue
        contig = f[0]
        strand = f[6]
        s = int(f[3]) - 1
        e = int(f[4])
        if tid in tx_extent:
            _, _, lo, hi = tx_extent[tid]
            tx_extent[tid] = (contig, strand, min(lo, s), max(hi, e))
        else:
            tx_extent[tid] = (contig, strand, s, e)

    genome = Fasta(str(genome_fa), as_raw=True, sequence_always_upper=True)
    out: dict[str, str] = {}
    for tid, (contig, strand, lo, hi) in tx_extent.items():
        if contig not in genome:
            out[tid] = "N" * k
            continue
        chrom_len = len(genome[contig])
        if strand == "+":
            up_lo = max(0, lo - k)
            seq = str(genome[contig][up_lo:lo])
            if len(seq) < k:
                seq = "N" * (k - len(seq)) + seq
        elif strand == "-":
            up_hi = min(chrom_len, hi + k)
            seq = str(genome[contig][hi:up_hi])
            if len(seq) < k:
                seq = seq + "N" * (k - len(seq))
            seq = _rev_comp(seq)
        else:
            seq = "N" * k
        seq = "".join(c if c in "ACGTN" else "N" for c in seq)
        out[tid] = seq
    return out


def _rewrite_with_prefix(
    in_fa: Path,
    out_fa: Path,
    prefixes: dict[str, str] | None,
    default: str,
) -> int:
    n = 0
    current_tid: str | None = None
    first_seq_line = True
    with open(in_fa) as fin, open(out_fa, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                fout.write(line)
                current_tid = line[1:].strip().split()[0]
                first_seq_line = True
                n += 1
            else:
                if first_seq_line:
                    prefix = default
                    if prefixes is not None and current_tid in prefixes:
                        prefix = prefixes[current_tid]
                    fout.write(prefix)
                    first_seq_line = False
                fout.write(line)
    return n


def _parse_tx_to_gene(gtf_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in Path(gtf_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        tid = _gtf_attr(f[8], "transcript_id")
        gid = _gtf_attr(f[8], "gene_id")
        if tid is not None and gid is not None and tid not in out:
            out[tid] = gid
    return out


def _filter_stringtie_gtf(
    in_path: Path,
    out_path: Path,
    min_cov: float | None,
    min_tpm: float | None,
    drop_unstranded: bool,
    drop_single_exon: bool,
) -> tuple[int, int]:
    """Filter a StringTie GTF by per-transcript metadata. Returns (kept, dropped)."""
    from collections import defaultdict

    tx_strand: dict[str, str] = {}
    tx_cov: dict[str, float] = {}
    tx_tpm: dict[str, float] = {}
    tx_exon_count: dict[str, int] = defaultdict(int)
    lines_by_tid: dict[str, list[str]] = defaultdict(list)
    header_lines: list[str] = []

    for raw in Path(in_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            header_lines.append(raw)
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        tid = _gtf_attr(f[8], "transcript_id")
        if tid is None:
            continue
        lines_by_tid[tid].append(raw)
        if f[2] == "transcript":
            tx_strand[tid] = f[6]
            cov = _gtf_attr(f[8], "cov")
            tpm = _gtf_attr(f[8], "TPM")
            if cov is not None:
                try:
                    tx_cov[tid] = float(cov)
                except ValueError:
                    pass
            if tpm is not None:
                try:
                    tx_tpm[tid] = float(tpm)
                except ValueError:
                    pass
        elif f[2] == "exon":
            tx_exon_count[tid] += 1
            tx_strand.setdefault(tid, f[6])

    keep: set[str] = set()
    for tid in lines_by_tid:
        if drop_unstranded and tx_strand.get(tid, ".") == ".":
            continue
        if min_cov is not None and tx_cov.get(tid, 0.0) < min_cov:
            continue
        if min_tpm is not None and tx_tpm.get(tid, 0.0) < min_tpm:
            continue
        if drop_single_exon and tx_exon_count.get(tid, 0) <= 1:
            continue
        keep.add(tid)

    with Path(out_path).open("w") as fh:
        for h in header_lines:
            fh.write(h + "\n")
        for tid, lines in lines_by_tid.items():
            if tid in keep:
                for ln in lines:
                    fh.write(ln + "\n")

    return len(keep), len(lines_by_tid) - len(keep)


def _encode_b2m_to_model(nuc_int, chunk_len: int):
    """Convert a b2m chunked nucleotide array (N, T) to model input (N, chunk_len, 6)."""
    import numpy as np

    N, T = nuc_int.shape
    if T > chunk_len:
        raise ValueError(
            f"b2m chunk T={T} exceeds model chunk_len={chunk_len}."
        )
    nuc = nuc_int.copy()
    pad_T = (nuc == -1)
    nuc[pad_T] = 4
    mask_lc = nuc > 4
    nuc[mask_lc] = nuc[mask_lc] - 5

    rows, cols = np.indices((N, T))
    oh_T = np.zeros((N, T, 6), dtype=np.float32)
    oh_T[rows, cols, nuc] = 1.0
    oh_T[pad_T, 4] = 0.0
    oh_T[pad_T, 5] = 1.0

    if T < chunk_len:
        tail = np.zeros((N, chunk_len - T, 6), dtype=np.float32)
        tail[..., 5] = 1.0
        oh = np.concatenate([oh_T, tail], axis=1)
        pad_mask = np.zeros((N, chunk_len), dtype=bool)
        pad_mask[:, :T] = pad_T
        pad_mask[:, T:] = True
    else:
        oh = oh_T
        pad_mask = pad_T
    return oh, pad_mask


# drusilla label index -> bricks2marble's 15-state index. b2m's
# `_split_regions` helper aggregates IR/I0..I2 to 0/1 and any other state
# to 2 (CDS); the boundary-mismatch detector in `_find_mismatches` treats
# {0,1,2,3} as "safe trailing states" and {4,5,6} as "exon at boundary".
# Mapping coding states onto E0/E1/E2 (and START/STOP onto Start/Stop)
# makes those checks behave correctly.
_TIB_TO_B2M = None  # built lazily to avoid importing numpy at module load


def _get_tib_to_b2m():
    global _TIB_TO_B2M
    if _TIB_TO_B2M is None:
        import numpy as np
        _TIB_TO_B2M = np.array([0, 7, 5, 6, 4, 14], dtype=np.int32)
    return _TIB_TO_B2M


def _make_predict_func(
    model,
    hmm,
    chunk_len: int,
    batch_size: int,
    label_store=None,
) -> Callable:
    import numpy as np
    from ..hmm.decode import viterbi_decode_batch

    tib_to_b2m = _get_tib_to_b2m()

    def predict_func(fasta):
        N, T = fasta.N, fasta.T
        if N == 0:
            return np.zeros((0, T), dtype=np.int32), None
        x, pad_mask = _encode_b2m_to_model(fasta.nuc, chunk_len)
        nuc_one_hot = x[..., :5]

        labels_out = np.empty((N, T), dtype=np.int32)
        for i in range(0, N, batch_size):
            sl = slice(i, i + batch_size)
            logits_b = model(x[sl], training=False).numpy()
            labels_b = viterbi_decode_batch(
                hmm, logits_b, nuc_one_hot[sl], pad_mask=pad_mask[sl],
            )
            labels_out[sl] = labels_b[:, :T]

        if label_store is not None:
            row = 0
            for seq in fasta:
                n = seq.N
                prev = label_store.get(seq.name)
                if prev is None or n > prev.shape[0]:
                    label_store[seq.name] = labels_out[row:row + n].copy()
                row += n

        return tib_to_b2m[labels_out].astype(np.int32), None

    return predict_func


def _resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve (weights, config) from --weights/--model/--config."""
    from .. import registry

    if args.weights is not None:
        if args.config is None:
            raise SystemExit(
                "--config is required when --weights is used (need architecture)."
            )
        return args.weights, args.config

    name = args.model or "vertebrates"
    weights = registry.resolve_weights(name)
    config = args.config if args.config is not None else registry.resolve_config(name)
    return weights, config


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import yaml

    weights_path, config_path = _resolve_model_paths(args)

    cfg = yaml.safe_load(open(config_path))
    dc, mc = cfg["data"], cfg["model"]
    chunk_len = dc["chunk_len"]

    # Resolve batch size (auto if not user-provided).
    if args.batch_size is None:
        from ..gpu import resolve_batch_size
        args.batch_size, bs_source = resolve_batch_size(chunk_len)
        print(f"[batch-size] {args.batch_size}  ({bs_source})", flush=True)
    else:
        print(f"[batch-size] {args.batch_size}  (user-supplied via --batch-size)",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Workdir: {args.out_dir}", flush=True)

    # 1. resolve StringTie GTF
    if args.bam is not None:
        stringtie_gtf = args.out_dir / "stringtie.gtf"
        print(f"Running StringTie on {args.bam} (longread={args.longread})",
              flush=True)
        st_cmd = ["stringtie", str(args.bam),
                  "-o", str(stringtie_gtf),
                  "-p", str(args.threads)]
        if args.longread:
            st_cmd.append("-L")
        _run_cmd(st_cmd)
    else:
        stringtie_gtf = args.stringtie_gtf

    # 1b. optional reference-free filtering of the StringTie GTF
    if (args.min_cov is not None or args.min_tpm is not None
            or args.drop_unstranded or args.drop_single_exon):
        filtered_gtf = args.out_dir / "stringtie.filtered.gtf"
        kept, dropped = _filter_stringtie_gtf(
            stringtie_gtf, filtered_gtf,
            min_cov=args.min_cov,
            min_tpm=args.min_tpm,
            drop_unstranded=args.drop_unstranded,
            drop_single_exon=args.drop_single_exon,
        )
        print(
            f"Filtered StringTie GTF (min_cov={args.min_cov}, "
            f"min_tpm={args.min_tpm}, drop_unstranded={args.drop_unstranded}, "
            f"drop_single_exon={args.drop_single_exon}): "
            f"kept {kept}, dropped {dropped} -> {filtered_gtf}",
            flush=True,
        )
        stringtie_gtf = filtered_gtf

    # 2. extract transcripts FASTA via gffread
    if args.transcripts_fa is None:
        transcripts_fa = args.out_dir / "transcripts.fa"
        print(f"Extracting transcripts -> {transcripts_fa}", flush=True)
        _run_cmd(["gffread", "-w", str(transcripts_fa),
                  "-g", str(args.genome), str(stringtie_gtf)])
    else:
        transcripts_fa = args.transcripts_fa

    orig_transcripts_fa = transcripts_fa

    # 2b. optional 5' padding.
    if args.prefix_pad_n > 0 and args.flank_bp > 0:
        raise SystemExit("--prefix-pad-n and --flank-bp are mutually exclusive")

    if args.prefix_pad_n > 0:
        K = args.prefix_pad_n
        padded_fa = args.out_dir / f"transcripts.pad{K}.fa"
        n_tx = _rewrite_with_prefix(transcripts_fa, padded_fa, None, "N" * K)
        print(f"Wrote 5'-padded FASTA (K={K}, n_tx={n_tx}) -> {padded_fa}",
              flush=True)
        transcripts_fa = padded_fa
    elif args.flank_bp > 0:
        K = args.flank_bp
        flank_fa = args.out_dir / f"transcripts.flank{K}.fa"
        print(f"Building 5'-flank prefixes (K={K}) from {args.genome}", flush=True)
        prefixes = _build_flank_prefixes(stringtie_gtf, args.genome, K)
        n_tx = _rewrite_with_prefix(transcripts_fa, flank_fa, prefixes, "N" * K)
        n_full = sum(1 for v in prefixes.values() if "N" not in v)
        print(
            f"Wrote 5'-flanked FASTA (K={K}, n_tx={n_tx}, "
            f"n_full_genomic_prefix={n_full}) -> {flank_fa}",
            flush=True,
        )
        transcripts_fa = flank_fa

    # 3. load model + HMM
    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    import bricks2marble as b2m
    from ..data.label_transcripts import parse_stringtie_gtf
    from ..data.tx_to_genome import project_tx_intervals_to_genomic
    from ..hmm.decode import build_decoder_hmm
    from ..model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=chunk_len)
    model.load_weights(str(weights_path))
    print(f"Loaded {mc['type']} weights from {weights_path}", flush=True)

    hmm = build_decoder_hmm(
        parallel=args.parallel,
        use_codon_emitter=not args.no_codon_emitter,
        restrict_start_to_ir_start=args.restrict_start_to_ir_start,
        ir_start_prior_ir=args.ir_prior,
    )
    print(
        f"Built OrfAnnotationHMM (parallel={args.parallel}, "
        f"codon_emitter={not args.no_codon_emitter}, "
        f"restrict_start_to_ir_start={args.restrict_start_to_ir_start}, "
        f"ir_prior={args.ir_prior})",
        flush=True,
    )

    # 4. exon structure for tx -> genome projection
    transcripts = parse_stringtie_gtf(stringtie_gtf)
    print(f"  transcripts (exon structure): {len(transcripts)}", flush=True)

    # 4b. Load transcript sequences for LORF classification (optional).
    tx_seqs: dict[str, str] = {}
    if args.lorf_class:
        from ..data.gtf_writer import classify_lorf  # noqa: F401
        tx_seqs = _load_tx_seqs(orig_transcripts_fa)
        print(f"Loaded {len(tx_seqs)} transcript sequences for LORF classification",
              flush=True)

    # 5. b2m predict / repredict adapters
    label_store: dict | None = (
        {} if (args.partial_out is not None or args.partial5_out is not None) else None
    )
    predict_func = _make_predict_func(
        model, hmm, chunk_len, args.batch_size, label_store=label_store,
    )
    repredict_func = predict_func

    intermediate_gtf = args.out_dir / "orfs_local.gtf"
    out_gtf = args.out_dir / "orfs.gtf"
    out_log = args.out_dir / "orfs.log"

    tx_to_gene: dict[str, str] = (
        _parse_tx_to_gene(stringtie_gtf) if args.single_isoform else {}
    )

    # Per input transcript, the outputs of postprocess:
    #   global_lines    genomic-coord CDS lines (one per exon-spanning segment)
    #   local_intervals transcript-coord ORF intervals (pad-stripped, half-open,
    #                   one entry per predicted ORF; typically length 1)
    #   coding_length   total predicted CDS length in nt
    #   lorf_class      LORF class if --lorf-class was set, else None
    per_tx_output: dict[str, dict] = {}
    lorf_counts: dict[str, int] = {}

    pad_k = max(args.prefix_pad_n, args.flank_bp)
    if args.flank_clip and args.flank_bp == 0:
        raise SystemExit("--flank-clip requires --flank-bp > 0")
    clip_in_pad = args.flank_clip
    n_unchanged = n_clipped = n_dropped_in_pad = 0

    def postprocess(_fasta, annot):
        nonlocal n_unchanged, n_clipped, n_dropped_in_pad
        if args.lorf_class:
            from ..data.gtf_writer import classify_lorf
        if args.min_coding_length > 0:
            b2m.tools.check_min_coding_length(
                annot, args.min_coding_length, remove=True,
            )
        for seq_ann in annot:
            for tx in seq_ann.transcripts():
                tid = tx.sequence
                if tid not in transcripts:
                    continue
                cds_intervals = [(c.start, c.end) for c in tx.cds]
                if not cds_intervals:
                    continue
                if pad_k > 0:
                    if cds_intervals[0][0] < pad_k:
                        if not clip_in_pad:
                            n_dropped_in_pad += 1
                            continue
                        s_orig = cds_intervals[0][0]
                        new_intervals: list[tuple[int, int]] = []
                        for s, e in cds_intervals:
                            if e <= pad_k:
                                continue
                            new_intervals.append((max(s, pad_k), e))
                        if not new_intervals:
                            n_dropped_in_pad += 1
                            continue
                        frame_shift = (3 - (new_intervals[0][0] - s_orig) % 3) % 3
                        if frame_shift > 0:
                            s0, e0 = new_intervals[0]
                            if s0 + frame_shift >= e0:
                                new_intervals.pop(0)
                            else:
                                new_intervals[0] = (s0 + frame_shift, e0)
                        if not new_intervals:
                            n_dropped_in_pad += 1
                            continue
                        cds_intervals = new_intervals
                        n_clipped += 1
                    else:
                        n_unchanged += 1
                    cds_intervals = [(s - pad_k, e - pad_k)
                                     for s, e in cds_intervals]
                orf_start = cds_intervals[0][0]
                orf_end   = cds_intervals[-1][1]
                tx_len    = transcripts[tid].length
                if args.min_utr_5 > 0 and orf_start < args.min_utr_5:
                    continue
                if args.min_utr_3 > 0 and (tx_len - orf_end) < args.min_utr_3:
                    continue
                coding_length = sum(e - s for s, e in cds_intervals)
                lc = (classify_lorf(tx_seqs[tid], orf_start)
                      if tx_seqs and tid in tx_seqs else None)
                if lc is not None:
                    lorf_counts[lc] = lorf_counts.get(lc, 0) + 1
                lines = project_tx_intervals_to_genomic(
                    tid, cds_intervals, transcripts[tid], "drusilla",
                    lorf_class=lc,
                )
                per_tx_output[tid] = {
                    "coding_length": coding_length,
                    "global_lines": lines,
                    "local_intervals": list(cds_intervals),
                    "lorf_class": lc,
                }
        return annot

    if intermediate_gtf.exists():
        intermediate_gtf.unlink()
    # b2m's _merge_replace_center assumes T is even (splits at T//2); pass
    # an even T_max so the repred-merge broadcast lines up.
    t_max_b2m = chunk_len - (chunk_len % 2)
    b2m.tools.annotate.annotate_genome(
        fasta=transcripts_fa,
        predict_func=predict_func,
        repredict_func=repredict_func,
        output=intermediate_gtf,
        log_file=out_log,
        model_name="drusilla",
        T_max=t_max_b2m,
        T_delta=0.1,
        min_sequence_size=1,
        reprediction_factor=0.5,
        concat_strand_to_reprediction=False,
        postprocess=postprocess,
        group_size_limit=1_000_000_000,
    )

    if args.single_isoform:
        best_per_gene: dict[str, tuple[int, str]] = {}
        for tid, entry in per_tx_output.items():
            length = entry["coding_length"]
            gid = tx_to_gene.get(tid, tid)
            cur = best_per_gene.get(gid)
            if cur is None or length > cur[0] or (length == cur[0] and tid < cur[1]):
                best_per_gene[gid] = (length, tid)
        keep_tids: set[str] = {tid for _, tid in best_per_gene.values()}
        print(
            f"single-isoform: kept {len(keep_tids)} of {len(per_tx_output)} "
            f"transcripts ({len(best_per_gene)} StringTie genes)",
            flush=True,
        )
    else:
        keep_tids = set(per_tx_output)

    # --- Subsequence-isoform collapse (default ON). --------------------
    # Runs on the genomic-coord predictions in keep_tids. Drops any
    # isoform whose spliced CDS is a (near-)sub-sequence of another
    # isoform in the same StringTie locus. Both orfs.gtf (global) and
    # orfs_local.gtf (transcript coords) are written from the same final
    # keep_tids set so the two files always agree.
    drop_map: dict[str, str] = {}
    if args.subseq_collapse and len(keep_tids) > 1:
        from ..postprocess.subseq_filter import find_dropable

        print(
            "[subseq-collapse] enabled by default. Drops predicted isoforms "
            "whose spliced CDS is a (near-)sub-sequence of another isoform "
            "in the same StringTie locus. Disable with --no-subseq-collapse.",
            flush=True,
        )
        print(
            f"[subseq-collapse] tolerances : "
            f"overhang={args.subseq_terminal_overhang_nt}nt "
            f"splice_shift={args.subseq_splice_shift_nt}nt "
            f"allow_exon_skip={args.subseq_allow_exon_skip}",
            flush=True,
        )

        # Build the by_tid dict for the collapse from the in-memory global
        # lines (avoids a round-trip through disk).
        by_tid: dict[str, dict] = {}
        for tid in keep_tids:
            if tid not in transcripts:
                continue
            tx = transcripts[tid]
            exons: list[tuple[int, int]] = []
            for line in per_tx_output[tid]["global_lines"]:
                f = line.split("\t")
                exons.append((int(f[3]) - 1, int(f[4])))
            exons.sort()
            by_tid[tid] = {
                "contig": tx.contig,
                "strand": tx.strand,
                "exons": exons,
                "lines": per_tx_output[tid]["global_lines"],
            }

        drop_map = find_dropable(
            by_tid,
            terminal_overhang_nt=args.subseq_terminal_overhang_nt,
            splice_shift_nt=args.subseq_splice_shift_nt,
            allow_exon_skip=args.subseq_allow_exon_skip,
        )
        n_in = len(by_tid)
        n_dropped = len(drop_map)
        n_kept = n_in - n_dropped
        pct = 100.0 * n_dropped / max(1, n_in)
        print(
            f"[subseq-collapse] input tx   : {n_in}\n"
            f"[subseq-collapse] dropped tx : {n_dropped} ({pct:.1f}%)\n"
            f"[subseq-collapse] kept tx    : {n_kept}",
            flush=True,
        )

        keep_tids -= set(drop_map)

    elif not args.subseq_collapse:
        print(
            "[subseq-collapse] disabled (--no-subseq-collapse). "
            "orfs.gtf / orfs_local.gtf contain all predicted isoforms.",
            flush=True,
        )

    # --- Write orfs.gtf (PRIMARY, global genomic coordinates). ---------
    n_lines = 0
    with open(out_gtf, "w") as fh:
        for tid in sorted(keep_tids):
            for line in per_tx_output[tid]["global_lines"]:
                fh.write(line + "\n")
                n_lines += 1
    print(
        f"[global]  wrote {n_lines} CDS lines from {len(keep_tids)} transcripts "
        f"-> {out_gtf}  (PRIMARY OUTPUT)",
        flush=True,
    )

    # --- Write orfs_local.gtf (transcript / local coordinates). --------
    # One CDS line per predicted ORF; contig column = StringTie
    # transcript_id, strand = "+" (transcripts are always in reading
    # direction after gffread -w), phase = 0 at the ORF start.
    local_gtf = args.out_dir / "orfs_local.gtf"
    n_local = 0
    with open(local_gtf, "w") as fh:
        for tid in sorted(keep_tids):
            entry = per_tx_output[tid]
            lc = entry.get("lorf_class")
            lorf_attr = f' lorf_class "{lc}";' if lc is not None else ""
            attrs = f'transcript_id "{tid}"; gene_id "{tid}";{lorf_attr}'
            for s, e in entry["local_intervals"]:
                if s >= e:
                    continue
                fh.write(
                    "\t".join([
                        tid, "drusilla", "CDS",
                        str(s + 1), str(e), ".", "+", "0",
                        attrs,
                    ]) + "\n"
                )
                n_local += 1
    print(
        f"[local]   wrote {n_local} CDS lines -> {local_gtf}",
        flush=True,
    )

    # Optional drop-report TSV.
    if args.subseq_report and args.subseq_collapse:
        report_tsv = args.out_dir / "orfs.dropped_subseq.tsv"
        with report_tsv.open("w", encoding="utf-8") as fh:
            fh.write("dropped_tid\tkeeper_tid\treason\n")
            for dropped, keeper in sorted(drop_map.items()):
                fh.write(f"{dropped}\t{keeper}\tsub_sequence_of_keeper\n")
        print(f"[subseq-collapse] drop report -> {report_tsv}", flush=True)
    if pad_k > 0:
        print(
            f"5'-prefix postprocess: unchanged={n_unchanged} "
            f"clipped_in_pad={n_clipped} dropped_in_pad={n_dropped_in_pad} "
            f"(clip_policy={'on' if clip_in_pad else 'off'})",
            flush=True,
        )
    if lorf_counts:
        for cls in ("LORF_UPSTOP", "sORF_UPSTOP", "upLORF", "LORF_NOUPSTOP"):
            print(f"  complete LORF {cls}: {lorf_counts.get(cls, 0)}", flush=True)

    # Partial ORF emission (3'-truncated).
    if args.partial_out is not None and label_store:
        from ..data.gtf_writer import extract_partial_orfs, classify_lorf
        n_partial = 0
        partial_lorf_counts: dict[str, int] = {}
        with open(args.partial_out, "w") as fh:
            for tid in sorted(label_store):
                if tid not in transcripts:
                    continue
                tx = transcripts[tid]
                chunk_lbl = label_store[tid]
                flat_lbl = chunk_lbl.ravel()
                real_len = tx.length + pad_k
                flat_lbl = flat_lbl[:real_len]
                if pad_k > 0:
                    flat_lbl = flat_lbl[pad_k:]
                for orf_start, orf_end in extract_partial_orfs(flat_lbl):
                    if args.min_coding_length > 0 and (orf_end - orf_start) < args.min_coding_length:
                        continue
                    if args.min_utr_5 > 0 and orf_start < args.min_utr_5:
                        continue
                    lc = (classify_lorf(tx_seqs[tid], orf_start)
                          if tx_seqs and tid in tx_seqs else None)
                    if lc is not None:
                        partial_lorf_counts[lc] = partial_lorf_counts.get(lc, 0) + 1
                    lines = project_tx_intervals_to_genomic(
                        tid, [(orf_start, orf_end)], tx, "drusilla",
                        lorf_class=lc,
                    )
                    for line in lines:
                        fh.write(line + "\n")
                    n_partial += 1
        print(f"Partial ORFs: {n_partial} -> {args.partial_out}", flush=True)
        if partial_lorf_counts:
            for cls in ("LORF_UPSTOP", "sORF_UPSTOP", "upLORF", "LORF_NOUPSTOP"):
                print(f"  partial LORF {cls}: {partial_lorf_counts.get(cls, 0)}",
                      flush=True)

    # 5'-partial ORF emission.
    if args.partial5_out is not None and label_store:
        from ..data.gtf_writer import extract_5prime_partial_orfs
        n_partial5 = 0
        with open(args.partial5_out, "w") as fh:
            for tid in sorted(label_store):
                if tid not in transcripts:
                    continue
                tx = transcripts[tid]
                chunk_lbl = label_store[tid]
                flat_lbl = chunk_lbl.ravel()
                real_len = tx.length + pad_k
                flat_lbl = flat_lbl[:real_len]
                if pad_k > 0:
                    flat_lbl = flat_lbl[pad_k:]
                for orf_start, orf_end in extract_5prime_partial_orfs(flat_lbl):
                    if args.min_coding_length > 0 and (orf_end - orf_start) < args.min_coding_length:
                        continue
                    lines = project_tx_intervals_to_genomic(
                        tid, [(orf_start, orf_end)], tx, "drusilla",
                    )
                    for line in lines:
                        fh.write(line + "\n")
                    n_partial5 += 1
        print(f"5'-partial ORFs: {n_partial5} -> {args.partial5_out}", flush=True)

    if not args.keep_tmp:
        for f in (intermediate_gtf, args.out_dir / "transcripts.fa",
                  args.out_dir / f"transcripts.pad{args.prefix_pad_n}.fa",
                  args.out_dir / f"transcripts.flank{args.flank_bp}.fa"):
            try:
                f.unlink()
            except (FileNotFoundError, OSError):
                pass
    return 0
