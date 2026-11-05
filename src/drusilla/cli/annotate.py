"""`drusilla annotate`: end-to-end ORF annotation from an assembly GTF or BAM.

Two input modes:
    --assembly-gtf <gtf> --genome <fa>       transcript assembly GTF (StringTie
                                              or any assembler writing exon lines
                                              with transcript_id / gene_id)
    --bam <bam>          --genome <fa>       run StringTie internally first
                                              (optionally after --fix-strand)

Model weights are resolved via the registry: the bundled ``drusilla_model``
is used by default, ``--model <name>`` selects another registered model
(the legacy per-clade models are downloaded on first use), and
``--weights <path>`` loads a locally-trained checkpoint.

Default output (in --out-dir):
    orfs.gtf        PRIMARY. Genomic-coordinate GTF of predicted CDS lines
                    (source = "drusilla"), 1-based inclusive, with the
                    subsequence-isoform collapse applied.
    orfs_local.gtf  Same predictions in transcript (local) coordinates:
                    one CDS line per predicted ORF, contig column holds
                    the assembler's transcript_id (``<tid>__rc`` for the
                    reverse-complement copy of an unstranded transcript).
    orfs.log        b2m annotation log.

Transcript handling (see ``drusilla.data.tx_augment``):
  * StringTie strand '.' transcripts are modelled on the forward strand
    (gffread extracts them as the forward genomic sequence). With
    --unstranded-both-strands the reverse complement is scanned as a
    second record ``<tid>__rc`` and the orientation with the longest ORF
    is kept (--unstranded-pick).
  * --flank-bp / --flank-3p-bp add real genomic context around every
    transcript. With --flank-keep an ORF whose START or STOP falls in a
    flank is completed into the genome and tagged with the GTF attribute
    ``completion "5p|3p|5p3p"`` (rescues 5'/3'-truncated assemblies).
  * --orfs-per-tx N emits up to N ORFs per transcript, longest first
    (ids ``<tid>``, ``<tid>.orf2``, ...; 0 = all). Needed for fused
    transcripts that carry several genes.

Subsequence collapse is ON by default. Disable with --no-subseq-collapse
(both orfs.gtf and orfs_local.gtf are then the raw predictions).
Opt into a drop-report TSV with --subseq-report.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from ..data import strand_fix_bam as _sfb

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


class _DeprecatedAlias(argparse.Action):
    """Store like a plain option but point the user at the new spelling."""

    def __call__(self, parser, namespace, values, option_string=None):
        new = "--" + self.dest.replace("_", "-")
        print(f"note: {option_string} is deprecated, use {new}",
              file=sys.stderr, flush=True)
        setattr(namespace, self.dest, values)


def add_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--assembly-gtf", type=Path, dest="assembly_gtf",
                     help="GTF of assembled transcripts: exon lines with "
                          "transcript_id and gene_id attributes, from StringTie "
                          "or any other assembler. StringTie is not run.")
    # Former spelling: hidden from --help, still accepted (Paludamentum
    # passes it) and warns on stderr.
    src.add_argument("--stringtie-gtf", type=Path, dest="assembly_gtf",
                     action=_DeprecatedAlias, help=argparse.SUPPRESS)
    src.add_argument("--bam", type=Path,
                     help="Sorted alignment BAM (run StringTie internally).")
    ap.add_argument("--transcripts-fa", type=Path, default=None,
                    help="FASTA of assembled transcripts (skips gffread).")
    ap.add_argument("--genome", type=Path, required=True,
                    help="Genome FASTA matching the GTF/BAM contig names.")

    # Model selection: a registered name (bundled, or downloaded for the
    # legacy models) or a local weights file (advanced / custom-trained).
    mdl = ap.add_mutually_exclusive_group()
    mdl.add_argument("--model", type=str, default=None,
                     help="Registered model name (see `drusilla models list`). "
                          f"Defaults to {DEFAULT_MODEL!r} if neither --model nor "
                          "--weights is given. A model manifest may preset the "
                          "unstranded / flank / orfs-per-tx options "
                          "(annotate_defaults); options given explicitly win.")
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

    # Context padding / genomic flanks.
    ap.add_argument("--prefix-pad-n", type=int, default=0,
                    help="Prepend this many literal 'N' bases as left-context. "
                         "ORFs starting inside the pad are dropped.")
    ap.add_argument("--flank-bp", type=int, default=None,
                    help="Prepend this many bp of real upstream genomic "
                         "context (RC for '-' strand), N-filled at contig "
                         "edges. Mutually exclusive with --prefix-pad-n. "
                         "By default ORFs whose START falls in the flank "
                         "are dropped (see --flank-clip / --flank-keep).")
    ap.add_argument("--flank-3p-bp", type=int, default=None,
                    help="Append this many bp of real downstream genomic "
                         "context (RC for '-' strand), N-filled at contig "
                         "edges. Only useful with --flank-keep: an ORF whose "
                         "STOP falls in the 3' flank is then emitted as a "
                         "completed CDS.")
    ap.add_argument("--flank-clip", action="store_true",
                    help="With --flank-bp: clip ORFs whose START is in the "
                         "flank to the first in-frame codon inside the tx "
                         "(emits a 5'-truncated CDS). Mutually exclusive "
                         "with --flank-keep.")
    ap.add_argument("--flank-keep", action=argparse.BooleanOptionalAction, default=None,
                    help="With --flank-bp/--flank-3p-bp: keep ORFs that "
                         "extend into the genomic flank(s) and project the "
                         "flank part to the genome ('ORF completion' for "
                         "5'/3'-truncated assemblies). Completed ORFs carry "
                         "the GTF attribute completion \"5p|3p|5p3p\". ORFs "
                         "with less than --flank-min-tx-frac of their coding "
                         "bases inside the assembled transcript are dropped.")
    ap.add_argument("--flank-min-tx-frac", type=float, default=0.5,
                    help="With --flank-keep: minimum fraction of an ORF's "
                         "coding bases that must lie inside the assembled "
                         "transcript (default 0.5).")

    # Unstranded transcripts / several ORFs per transcript.
    # The three options below and the flank options default to None so that a
    # model manifest can preset them (annotate_defaults); see
    # _apply_inference_defaults for the built-in fallbacks.
    ap.add_argument("--unstranded-both-strands",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="For transcripts whose assembler strand is '.' "
                         "(gffread extracts the forward genomic sequence) "
                         "also scan the reverse complement as a separate "
                         "record '<tid>__rc' with a '-' strand model. "
                         "Without this, minus-strand genes on unstranded "
                         "(mostly single-exon) transcripts cannot be called. "
                         "Built-in default off; the drusilla_model manifest "
                         "turns it on (--no-unstranded-both-strands to override).")
    ap.add_argument("--unstranded-pick", choices=("nonoverlap", "longest", "both"),
                    default=None,
                    help="With --unstranded-both-strands: 'nonoverlap' keeps "
                         "the ORFs of both orientations, longest first, "
                         "dropping any ORF that overlaps a kept one on the "
                         "genome (matches label policy v2 of drusilla_model); "
                         "'longest' keeps only the orientation with the "
                         "longest ORF (built-in default); 'both' keeps "
                         "everything.")
    ap.add_argument("--orfs-per-tx", type=int, default=None,
                    help="Maximum number of ORFs emitted per transcript, "
                         "longest first (built-in default 1; 0 = no limit, the "
                         "drusilla_model manifest default, as its training "
                         "labels hold every gene of a fused transcript). Extra "
                         "ORFs get transcript_id '<tid>.orf2', '<tid>.orf3', ...")

    ap.add_argument("--threads", type=int, default=4,
                    help="Threads for StringTie and for BAM (de)compression "
                         "under --fix-strand (gffread is single-threaded).")
    ap.add_argument("--longread", action="store_true",
                    help="Pass -L to StringTie (long-read assembly mode).")
    ap.add_argument("--fix-strand", action="store_true",
                    help="With --bam: recover per-run library strandedness "
                         "first (XS tags on all reads of stranded runs; see "
                         "`drusilla strand-fix-bam`) and run StringTie on the "
                         "fixed BAM. Needs pysam. Writes strand_runs.tsv.")
    ap.add_argument("--strand-run-regex", default=_sfb.DEFAULT_RUN_REGEX,
                    help="With --fix-strand: regex whose group 1 is the run "
                         "id in a read name (default: leading letters+digits, "
                         "e.g. SRR123456); 'none': the BAM is one library.")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Do not delete intermediate FASTA / BAM files and "
                         "b2m's own GTF in --out-dir.")

    # Reference-free assembly pre-filters (cov / TPM are StringTie attributes).
    ap.add_argument("--min-cov", type=float, default=None,
                    help="Drop transcripts whose cov attribute (StringTie) "
                         "is < X; a transcript without cov counts as 0.")
    ap.add_argument("--min-tpm", type=float, default=None,
                    help="Drop transcripts whose TPM attribute (StringTie) "
                         "is < X; a transcript without TPM counts as 0.")
    ap.add_argument("--drop-unstranded", action="store_true",
                    help="Drop transcripts with strand '.'.")
    ap.add_argument("--drop-single-exon", action="store_true",
                    help="Drop single-exon transcripts.")

    # Post-decode filters.
    ap.add_argument("--min-coding-length", type=int, default=200,
                    help="Drop predicted ORFs with total CDS length < X nt "
                         "(default 200). Set to 0 to disable.")
    ap.add_argument("--single-isoform", action="store_true",
                    help="Keep at most one transcript per StringTie gene_id "
                         "(longest predicted CDS wins).")
    ap.add_argument("--min-utr-5", type=int, default=0,
                    help="Drop ORFs whose START is within N nt of the tx 5' end "
                         "(not applied to ends completed into a flank).")
    ap.add_argument("--min-utr-3", type=int, default=0,
                    help="Drop ORFs whose STOP is within N nt of the tx 3' end "
                         "(not applied to ends completed into a flank).")
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


def _filter_assembly_gtf(
    in_path: Path,
    out_path: Path,
    min_cov: float | None,
    min_tpm: float | None,
    drop_unstranded: bool,
    drop_single_exon: bool,
) -> tuple[int, int]:
    """Filter an assembly GTF by StringTie-style per-transcript attributes
    (cov, TPM), strand and exon count. Returns (kept, dropped)."""
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


# Model used when neither --model nor --weights is given: the bundled
# multi-clade model (model_cfg/drusilla_model.yaml, weights in the package).
# The legacy per-clade models stay available by name (downloaded on use).
DEFAULT_MODEL = "drusilla_model"

# Inference options a model manifest may preset in its ``annotate_defaults``
# block (keys = argparse dests). Precedence: command line > manifest > these
# built-ins, which reproduce the behaviour of Drusilla <= 0.2.
INFERENCE_DEFAULTS: dict[str, object] = {
    "unstranded_both_strands": False,
    "unstranded_pick": "longest",
    "orfs_per_tx": 1,
    "flank_bp": 0,
    "flank_3p_bp": 0,
    "flank_keep": False,
}
_UNSTRANDED_PICKS = ("nonoverlap", "longest", "both")


def _manifest_inference_defaults(args: argparse.Namespace) -> dict[str, object]:
    """``annotate_defaults`` of the registered model that will be used, read
    from its manifest without downloading anything. Empty for ``--weights``
    runs and for unknown models (``_resolve_model_paths`` reports those)."""
    if args.weights is not None:
        return {}
    from .. import registry
    try:
        data = registry.load_manifest(args.model or DEFAULT_MODEL).data
    except registry.RegistryError:
        return {}
    block = data.get("annotate_defaults") or {}
    if not isinstance(block, dict):
        raise SystemExit(f"model manifest {data.get('name')!r}: annotate_defaults must be a mapping")
    unknown = sorted(set(block) - set(INFERENCE_DEFAULTS))
    if unknown:
        raise SystemExit(f"model manifest {data.get('name')!r}: unknown annotate_defaults {unknown}; "
                         f"allowed: {sorted(INFERENCE_DEFAULTS)}")
    return dict(block)


def _apply_inference_defaults(args: argparse.Namespace,
                              manifest_defaults: dict[str, object] | None = None) -> list[str]:
    """Fill the options left unset on the command line (None) from the
    manifest block, then from ``INFERENCE_DEFAULTS``. Idempotent. Returns the
    option names that came from the manifest."""
    from_manifest: list[str] = []
    for key, builtin in INFERENCE_DEFAULTS.items():
        if getattr(args, key, None) is not None:
            continue
        if manifest_defaults and key in manifest_defaults:
            value = manifest_defaults[key]
            if isinstance(builtin, bool):
                if not isinstance(value, bool):
                    raise SystemExit(f"annotate_defaults.{key} must be true/false, got {value!r}")
            elif isinstance(builtin, int):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise SystemExit(f"annotate_defaults.{key} must be an integer >= 0, got {value!r}")
            elif key == "unstranded_pick" and value not in _UNSTRANDED_PICKS:
                raise SystemExit(f"annotate_defaults.unstranded_pick must be one of "
                                 f"{_UNSTRANDED_PICKS}, got {value!r}")
            setattr(args, key, value)
            from_manifest.append(key)
        else:
            setattr(args, key, builtin)
    return from_manifest


def _resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve (weights, arch_config) from --weights/--model/--config.

    Precedence:
      * ``--weights FILE`` + ``--config FILE`` — local override (advanced use).
        Both are required together; the arch config travels with released
        models, so a local checkpoint needs its own.
      * ``--model NAME`` (or the default ``drusilla_model``): resolve via the
        registry. A bundled model is read from the package; a legacy model
        is downloaded and extracted on first use.
    """
    from .. import registry

    if args.weights is not None:
        if args.config is None:
            raise SystemExit(
                "--config is required when --weights is used "
                "(the architecture config is not carried in .weights.h5)."
            )
        return args.weights, args.config

    name = args.model or DEFAULT_MODEL
    resolved = registry.resolve_model(name)
    # --config is not needed for a registered model (the registry provides
    # its arch config). Warn if user passed both, use the local one.
    if args.config is not None:
        print(
            f"note: --config {args.config} overrides the arch config that "
            f"ships with model {name!r}.",
            flush=True,
        )
        arch_config = args.config
    else:
        arch_config = resolved.arch_config_path
    print(
        f"[model] {resolved.name} v{resolved.version}  "
        f"weights={resolved.weights_path}  arch={arch_config}",
        flush=True,
    )
    return resolved.weights_path, arch_config


def _check_option_consistency(args: argparse.Namespace) -> None:
    # Options a manifest may preset are None until resolved; fall back to the
    # built-ins here so the checks also work on a freshly parsed namespace.
    _apply_inference_defaults(args)
    if args.prefix_pad_n > 0 and (args.flank_bp > 0 or args.flank_3p_bp > 0):
        raise SystemExit("--prefix-pad-n and --flank-bp/--flank-3p-bp are mutually exclusive")
    if args.flank_clip and args.flank_bp == 0:
        raise SystemExit("--flank-clip requires --flank-bp > 0")
    if args.flank_keep and args.flank_bp == 0 and args.flank_3p_bp == 0:
        raise SystemExit("--flank-keep requires --flank-bp and/or --flank-3p-bp > 0")
    if args.flank_keep and args.flank_clip:
        raise SystemExit("--flank-keep and --flank-clip are mutually exclusive")
    if args.fix_strand and args.bam is None:
        raise SystemExit("--fix-strand requires --bam (StringTie runs on the fixed BAM)")
    if args.fix_strand and importlib.util.find_spec("pysam") is None:
        raise SystemExit("--fix-strand needs pysam: pip install pysam  (or drusilla[bam])")
    if args.orfs_per_tx < 0:
        raise SystemExit("--orfs-per-tx must be >= 0")
    if not 0.0 <= args.flank_min_tx_frac <= 1.0:
        raise SystemExit("--flank-min-tx-frac must be between 0 and 1")


def run(args: argparse.Namespace) -> int:
    from collections import Counter

    import numpy as np
    import yaml

    from ..data import tx_augment as txa
    from ..data.label_transcripts import parse_stringtie_gtf
    from ..data.tx_to_genome import genomic_segments, project_tx_intervals_to_genomic

    from_manifest = _apply_inference_defaults(args, _manifest_inference_defaults(args))
    _check_option_consistency(args)
    weights_path, config_path = _resolve_model_paths(args)
    if from_manifest:
        print("[model] inference defaults taken from the model manifest: "
              + ", ".join(f"{k}={getattr(args, k)}" for k in from_manifest), flush=True)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
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
    tmp_files: list[Path] = []

    # 1. resolve the transcript assembly GTF
    if args.bam is not None:
        bam = args.bam
        if args.fix_strand:
            from ..data import strand_fix_bam as sfb
            sfix_bam = args.out_dir / "aligned.strandfix.bam"
            report = args.out_dir / "strand_runs.tsv"
            print(f"Recovering per-run library strandedness -> {sfix_bam}", flush=True)
            summary = sfb.fix_strand(
                bam, sfix_bam, report,
                run_regex=args.strand_run_regex,
                threads=args.threads, index=False,
            )
            print(
                f"strand-fix: {summary['stranded_runs']} of {summary['runs']} runs "
                f"stranded, {summary['tagged']} of {summary['reads']} reads tagged "
                f"-> {report}",
                flush=True,
            )
            bam = sfix_bam
            tmp_files.append(sfix_bam)
        assembly_gtf = args.out_dir / "stringtie.gtf"
        print(f"Running StringTie on {bam} (longread={args.longread})", flush=True)
        st_cmd = ["stringtie", str(bam),
                  "-o", str(assembly_gtf),
                  "-p", str(args.threads)]
        if args.longread:
            st_cmd.append("-L")
        _run_cmd(st_cmd)
    else:
        assembly_gtf = args.assembly_gtf

    # 1b. optional reference-free filtering of the assembly GTF
    if (args.min_cov is not None or args.min_tpm is not None
            or args.drop_unstranded or args.drop_single_exon):
        filtered_gtf = args.out_dir / "stringtie.filtered.gtf"
        kept, dropped = _filter_assembly_gtf(
            assembly_gtf, filtered_gtf,
            min_cov=args.min_cov,
            min_tpm=args.min_tpm,
            drop_unstranded=args.drop_unstranded,
            drop_single_exon=args.drop_single_exon,
        )
        print(
            f"Filtered assembly GTF (min_cov={args.min_cov}, "
            f"min_tpm={args.min_tpm}, drop_unstranded={args.drop_unstranded}, "
            f"drop_single_exon={args.drop_single_exon}): "
            f"kept {kept}, dropped {dropped} -> {filtered_gtf}",
            flush=True,
        )
        assembly_gtf = filtered_gtf

    # 2. extract transcripts FASTA via gffread
    if args.transcripts_fa is None:
        transcripts_fa = args.out_dir / "transcripts.fa"
        print(f"Extracting transcripts -> {transcripts_fa}", flush=True)
        _run_cmd(["gffread", "-w", str(transcripts_fa),
                  "-g", str(args.genome), str(assembly_gtf)])
        tmp_files.append(transcripts_fa)
    else:
        transcripts_fa = args.transcripts_fa

    # 2a. exon structure for tx-coord -> genomic-coord projection. Strand
    # '.' transcripts are normalised to '+' here because gffread emits
    # them as the forward genomic sequence.
    transcripts = parse_stringtie_gtf(assembly_gtf)
    n_unstranded = sum(1 for t in transcripts.values() if t.unstranded)
    print(f"  transcripts (exon structure): {len(transcripts)} "
          f"({n_unstranded} unstranded)", flush=True)

    # 2b. unstranded transcripts: also scan the reverse complement. The RC
    # copy is a separate record '<tid>__rc' with a '-' strand model.
    rc_base_tids: list[str] = []
    if args.unstranded_both_strands and n_unstranded > 0:
        transcripts, rc_base_tids = txa.augment_unstranded(transcripts)
        aug_fa = args.out_dir / "transcripts.bothstrands.fa"
        n_rec = txa.rewrite_fasta(
            transcripts_fa, aug_fa, txa.with_rc_copies(set(rc_base_tids)),
        )
        print(
            f"Added reverse-complement copies for {len(rc_base_tids)} "
            f"unstranded transcripts -> {aug_fa} ({n_rec} records)",
            flush=True,
        )
        transcripts_fa = aug_fa
        tmp_files.append(aug_fa)

    # Unpadded FASTA (RC copies included) for LORF scanning.
    orig_transcripts_fa = transcripts_fa

    # 2c. optional padding. Three modes:
    #   --prefix-pad-n K                  : prepend K literal Ns.
    #   --flank-bp K5 [--flank-3p-bp K3]  : real genomic context on the 5'
    #                                       (and 3') side, RC for '-' strand,
    #                                       N-filled where a contig edge cuts
    #                                       the flank short.
    # Every padded record has length K5 + tx_len + K3. ORFs are classified
    # against that layout in postprocess (see tx_augment.place_orf).
    flank_info: dict[str, txa.FlankInfo] = {}
    k5 = args.prefix_pad_n if args.prefix_pad_n > 0 else args.flank_bp
    k3 = 0 if args.prefix_pad_n > 0 else args.flank_3p_bp
    if k5 > 0 or k3 > 0:
        if args.prefix_pad_n > 0:
            flank_info = {tid: txa.n_pad_only(tx, k5) for tid, tx in transcripts.items()}
            padded_fa = args.out_dir / f"transcripts.pad{k5}.fa"
        else:
            print(f"Building genomic flanks (5p={k5}, 3p={k3}) from {args.genome} ...",
                  flush=True)
            genome = txa.FaidxGenome(args.genome)
            flank_info = {
                tid: txa.build_flank(tx, genome, k5, k3)
                for tid, tx in transcripts.items()
            }
            padded_fa = args.out_dir / f"transcripts.flank{k5}_{k3}.fa"

        def _pad(name: str, seq: str) -> list[tuple[str, str]]:
            fi = flank_info.get(name)
            if fi is None:      # not in the GTF: plain N padding keeps coordinates aligned
                return [(name, "N" * k5 + seq + "N" * k3)]
            if len(seq) != fi.tx_len:
                # Exon model and extracted sequence disagree (odd GTF record):
                # fall back to plain N padding with the true length so ORFs in
                # the pads are still dropped and coordinates stay aligned.
                print(f"[warn] {name}: FASTA length {len(seq)} != exon model "
                      f"length {fi.tx_len}; N-padding without genomic flank",
                      flush=True)
                flank_info[name] = txa.FlankInfo(k5=k5, k3=k3, prefix="", suffix="",
                                                 model=transcripts[name],
                                                 tx_len=len(seq))
                return [(name, "N" * k5 + seq + "N" * k3)]
            return [(name, fi.padded_record(seq))]

        n_tx = txa.rewrite_fasta(transcripts_fa, padded_fa, _pad)
        n_full = sum(1 for fi in flank_info.values() if fi.n_pad5 == 0 and fi.n_pad3 == 0)
        print(
            f"Wrote padded FASTA (k5={k5}, k3={k3}, n_tx={n_tx}, "
            f"n_full_genomic_flank={n_full}) -> {padded_fa}",
            flush=True,
        )
        transcripts_fa = padded_fa
        tmp_files.append(padded_fa)

    # 3. load model + HMM
    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    import bricks2marble as b2m
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

    # 4. Load transcript sequences for LORF classification (optional).
    tx_seqs: dict[str, str] = {}
    classify_lorf = None
    if args.lorf_class:
        from ..data.gtf_writer import classify_lorf
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

    # 6. annotate transcripts as a "genome" and collect every decoded
    # transcript-coord ORF via the postprocess hook. Projection to genomic
    # CDS lines happens after decoding so several ORFs per transcript can
    # be ranked and given ids. b2m's own GTF is a by-product we delete.
    b2m_gtf = args.out_dir / "orfs.b2m_local.gtf"
    out_gtf = args.out_dir / "orfs.gtf"
    local_gtf = args.out_dir / "orfs_local.gtf"
    out_log = args.out_dir / "orfs.log"

    tx_to_gene: dict[str, str] = (
        _parse_tx_to_gene(assembly_gtf) if args.single_isoform else {}
    )

    # record name (tid or tid__rc) -> list of ORF dicts:
    #   coding_length  total predicted CDS length in nt
    #   intervals      CDS intervals in the coordinates of `model`
    #   model          transcript proper, or the flank-extended model when
    #                  the ORF was completed into a flank
    #   tx_start/end   ORF bounds relative to the transcript proper (may lie
    #                  outside [0, tx_len) for completed ORFs)
    #   completion     None / "5p" / "3p" / "5p3p"
    #   lorf_class     LORF class if --lorf-class was set, else None
    per_tx_orfs: dict[str, list[dict]] = {}
    lorf_counts: dict[str, int] = {}
    counters: Counter = Counter()

    def postprocess(_fasta, annot):
        if args.min_coding_length > 0:
            b2m.tools.check_min_coding_length(
                annot, args.min_coding_length, remove=True,
            )
        for seq_ann in annot:
            # `_all` holds every decoded transcript (several ORFs per
            # sequence); fall back to the public iterator on older b2m.
            txs = getattr(seq_ann, "_all", None)
            if txs is None:
                txs = seq_ann.transcripts()
            for tx in txs:
                tid = tx.sequence
                if tid not in transcripts:
                    continue
                cds = sorted((c.start, c.end) for c in tx.cds)
                if not cds:
                    continue
                tx_len = transcripts[tid].length
                res = txa.resolve_orf(
                    cds, transcripts[tid], flank_info.get(tid),
                    keep=args.flank_keep, clip=args.flank_clip,
                    min_tx_frac=args.flank_min_tx_frac,
                )
                if isinstance(res, str):
                    counters[res] += 1
                    continue
                counters[res.outcome] += 1
                completion = res.completion
                if completion is not None:
                    if "5p" in completion:
                        counters["completed_5p"] += 1
                    if "3p" in completion:
                        counters["completed_3p"] += 1

                # Reference-free proxy for truncated assemblies: skip ORFs
                # whose START or STOP sits flush against a transcript end.
                # Not applied to an end that was completed into a flank.
                if completion is None or "5p" not in completion:
                    if args.min_utr_5 > 0 and res.tx_start < args.min_utr_5:
                        continue
                if completion is None or "3p" not in completion:
                    if args.min_utr_3 > 0 and (tx_len - res.tx_end) < args.min_utr_3:
                        continue
                coding_length = sum(e - s for s, e in res.intervals)
                # b2m applied --min-coding-length to the padded record;
                # --flank-clip can shorten an ORF below it afterwards.
                if args.min_coding_length > 0 and coding_length < args.min_coding_length:
                    counters["dropped_short_after_clip"] += 1
                    continue
                lc = None
                if classify_lorf is not None and tid in tx_seqs and res.tx_start >= 0:
                    lc = classify_lorf(tx_seqs[tid], res.tx_start)
                    lorf_counts[lc] = lorf_counts.get(lc, 0) + 1
                per_tx_orfs.setdefault(tid, []).append({
                    "coding_length": coding_length,
                    "intervals": res.intervals,
                    "model": res.model,
                    "tx_start": res.tx_start,
                    "tx_end": res.tx_end,
                    "completion": completion,
                    "lorf_class": lc,
                })
        return annot

    if b2m_gtf.exists():
        b2m_gtf.unlink()
    # b2m's _merge_replace_center assumes T is even (splits at T//2); pass
    # an even T_max so the repred-merge broadcast lines up.
    t_max_b2m = chunk_len - (chunk_len % 2)
    b2m.tools.annotate.annotate_genome(
        fasta=transcripts_fa,
        predict_func=predict_func,
        repredict_func=repredict_func,
        output=b2m_gtf,
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
    tmp_files.append(b2m_gtf)

    # 7a. unstranded transcripts, --unstranded-pick nonoverlap: pool the ORFs
    # of both orientations and keep them longest first, dropping any ORF that
    # shares genomic bases with a kept one (a gene called in both orientations
    # keeps the longer call; genes on different strands of one transcript
    # survive). This is the inference rule matching label policy v2, whose
    # training labels hold both orientations of every unstranded transcript.
    if rc_base_tids and args.unstranded_pick == "nonoverlap":
        n_fwd = n_rc = n_dropped = 0
        for b in rc_base_tids:
            fwd = per_tx_orfs.get(b, [])
            rc = per_tx_orfs.get(b + txa.RC_SUFFIX, [])
            if not fwd and not rc:
                continue
            pool = [(o, b) for o in fwd] + [(o, b + txa.RC_SUFFIX) for o in rc]
            keep = txa.select_nonoverlapping(
                [(o["coding_length"], genomic_segments(o["intervals"], o["model"]))
                 for o, _ in pool],
            )
            n_dropped += len(pool) - len(keep)
            for tid in (b, b + txa.RC_SUFFIX):
                kept = [pool[i][0] for i in keep if pool[i][1] == tid]
                if kept:
                    per_tx_orfs[tid] = kept
                else:
                    per_tx_orfs.pop(tid, None)
            n_fwd += sum(pool[i][1] == b for i in keep)
            n_rc += len(keep) - sum(pool[i][1] == b for i in keep)
        print(f"unstranded transcripts: ORFs kept forward {n_fwd}, "
              f"reverse-complement {n_rc}; dropped as genomic overlaps "
              f"{n_dropped}", flush=True)

    # --unstranded-pick longest: keep the orientation with the longest ORF
    # ('both' keeps everything).
    if rc_base_tids and args.unstranded_pick == "longest":
        n_fwd = n_rc = 0
        for b in rc_base_tids:
            fwd = per_tx_orfs.get(b)
            rc = per_tx_orfs.get(b + txa.RC_SUFFIX)
            if not fwd and not rc:
                continue
            choice = txa.pick_orientation(
                max(o["coding_length"] for o in fwd) if fwd else None,
                max(o["coding_length"] for o in rc) if rc else None,
            )
            if choice == "rc":
                per_tx_orfs.pop(b, None)
                n_rc += 1
            else:
                per_tx_orfs.pop(b + txa.RC_SUFFIX, None)
                n_fwd += 1
        print(f"unstranded transcripts with ORFs: forward kept {n_fwd}, "
              f"reverse-complement kept {n_rc}", flush=True)

    # 7b. rank ORFs per transcript (longest first) and cap their number.
    n_extra = 0
    for orfs in per_tx_orfs.values():
        orfs.sort(key=lambda o: -o["coding_length"])
        if args.orfs_per_tx > 0 and len(orfs) > args.orfs_per_tx:
            del orfs[args.orfs_per_tx:]
        n_extra += len(orfs) - 1

    # 7c. With --single-isoform, group by the StringTie gene_id and keep the
    # transcript with the longest predicted CDS (ties broken by id).
    if args.single_isoform:
        best_per_gene: dict[str, tuple[int, str]] = {}
        for tid, orfs in per_tx_orfs.items():
            length = orfs[0]["coding_length"]
            base = txa.base_tid(tid)
            gid = tx_to_gene.get(base, base)
            cur = best_per_gene.get(gid)
            if cur is None or length > cur[0] or (length == cur[0] and tid < cur[1]):
                best_per_gene[gid] = (length, tid)
        keep_tx: set[str] = {tid for _, tid in best_per_gene.values()}
        print(
            f"single-isoform: kept {len(keep_tx)} of {len(per_tx_orfs)} "
            f"transcripts ({len(best_per_gene)} StringTie genes)",
            flush=True,
        )
    else:
        keep_tx = set(per_tx_orfs)

    # 7d. one record per ORF id: genomic CDS lines + local bounds.
    orf_records: dict[str, dict] = {}
    for tid in sorted(keep_tx):
        orfs = per_tx_orfs[tid]
        for oid, o in zip(txa.assign_orf_ids(tid, len(orfs)), orfs):
            lines = project_tx_intervals_to_genomic(
                oid, o["intervals"], o["model"], "drusilla",
                lorf_class=o["lorf_class"],
            )
            if o["completion"] is not None:
                lines = [ln + f' completion "{o["completion"]}";' for ln in lines]
            orf_records[oid] = {
                "tid": tid,
                "model": o["model"],
                "global_lines": lines,
                "tx_start": o["tx_start"],
                "tx_end": o["tx_end"],
                "completion": o["completion"],
                "lorf_class": o["lorf_class"],
            }
    keep_ids: set[str] = set(orf_records)

    # --- Subsequence-isoform collapse (default ON). --------------------
    # Runs on the genomic-coord predictions. Drops any ORF whose spliced
    # CDS is a (near-)sub-sequence of another ORF in the same StringTie
    # locus (ORF ids are mapped back to their locus, see
    # subseq_filter.gene_id_from_tx). orfs.gtf and orfs_local.gtf are both
    # written from the final keep set so the two files always agree.
    drop_map: dict[str, str] = {}
    if args.subseq_collapse and len(keep_ids) > 1:
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
        by_tid: dict[str, dict] = {}
        for oid in keep_ids:
            rec = orf_records[oid]
            exons: list[tuple[int, int]] = []
            for line in rec["global_lines"]:
                f = line.split("\t")
                exons.append((int(f[3]) - 1, int(f[4])))
            exons.sort()
            by_tid[oid] = {
                "contig": rec["model"].contig,
                "strand": rec["model"].strand,
                "exons": exons,
                "lines": rec["global_lines"],
            }
        drop_map = find_dropable(
            by_tid,
            terminal_overhang_nt=args.subseq_terminal_overhang_nt,
            splice_shift_nt=args.subseq_splice_shift_nt,
            allow_exon_skip=args.subseq_allow_exon_skip,
        )
        n_in = len(by_tid)
        n_dropped = len(drop_map)
        pct = 100.0 * n_dropped / max(1, n_in)
        print(
            f"[subseq-collapse] input ORFs : {n_in}\n"
            f"[subseq-collapse] dropped    : {n_dropped} ({pct:.1f}%)\n"
            f"[subseq-collapse] kept       : {n_in - n_dropped}",
            flush=True,
        )
        keep_ids -= set(drop_map)
    elif not args.subseq_collapse:
        print(
            "[subseq-collapse] disabled (--no-subseq-collapse). "
            "orfs.gtf / orfs_local.gtf contain all predicted isoforms.",
            flush=True,
        )

    # --- Write orfs.gtf (PRIMARY, global genomic coordinates). ---------
    n_lines = 0
    completion_counts: Counter = Counter()
    kept_tx: set[str] = set()
    with open(out_gtf, "w") as fh:
        for oid in sorted(keep_ids):
            rec = orf_records[oid]
            kept_tx.add(rec["tid"])
            if rec["completion"] is not None:
                completion_counts[rec["completion"]] += 1
            for line in rec["global_lines"]:
                fh.write(line + "\n")
                n_lines += 1
    print(
        f"[global]  wrote {n_lines} CDS lines / {len(keep_ids)} ORFs from "
        f"{len(kept_tx)} transcripts -> {out_gtf}  (PRIMARY OUTPUT; extra ORFs "
        f"beyond the first per transcript: {n_extra}, cap={args.orfs_per_tx or 'none'})",
        flush=True,
    )

    # --- Write orfs_local.gtf (transcript / local coordinates). --------
    # One CDS line per ORF; contig column = sequence record (StringTie
    # transcript_id, or '<tid>__rc' for a reverse-complement copy), strand
    # '+'. The part of a completed ORF that lies in a genomic flank is
    # outside the transcript sequence and is clipped here (phase then
    # reflects the clipped bases); the `completion` attribute marks such ORFs.
    n_local = 0
    with open(local_gtf, "w") as fh:
        for oid in sorted(keep_ids):
            rec = orf_records[oid]
            tx_len = transcripts[rec["tid"]].length
            s, e = max(0, rec["tx_start"]), min(tx_len, rec["tx_end"])
            if s >= e:
                continue
            attrs = f'transcript_id "{oid}"; gene_id "{oid}";'
            if rec["lorf_class"] is not None:
                attrs += f' lorf_class "{rec["lorf_class"]}";'
            if rec["completion"] is not None:
                attrs += f' completion "{rec["completion"]}";'
            # phase of the first written base: non-zero only when the ORF
            # start lies in the 5' flank and the line was clipped to position 0.
            phase = (3 - (s - rec["tx_start"]) % 3) % 3
            fh.write("\t".join([
                rec["tid"], "drusilla", "CDS", str(s + 1), str(e), ".", "+", str(phase), attrs,
            ]) + "\n")
            n_local += 1
    print(f"[local]   wrote {n_local} CDS lines -> {local_gtf}", flush=True)

    # Optional drop-report TSV.
    if args.subseq_report and args.subseq_collapse:
        report_tsv = args.out_dir / "orfs.dropped_subseq.tsv"
        with report_tsv.open("w", encoding="utf-8") as fh:
            fh.write("dropped_tid\tkeeper_tid\treason\n")
            for dropped, keeper in sorted(drop_map.items()):
                fh.write(f"{dropped}\t{keeper}\tsub_sequence_of_keeper\n")
        print(f"[subseq-collapse] drop report -> {report_tsv}", flush=True)
    if flank_info:
        print("padding postprocess: " + ", ".join(
            f"{k}={counters[k]}" for k in sorted(counters)), flush=True)
        if completion_counts:
            print("ORF completion into genomic flank: " + ", ".join(
                f"{k}={v}" for k, v in sorted(completion_counts.items())), flush=True)
    if lorf_counts:
        for cls in ("LORF_UPSTOP", "sORF_UPSTOP", "upLORF", "LORF_NOUPSTOP"):
            print(f"  complete LORF {cls}: {lorf_counts.get(cls, 0)}", flush=True)

    def _tx_labels(tid: str, tx) -> np.ndarray:
        """Flat 6-state labels for the transcript proper (padding stripped)."""
        flat = label_store[tid].ravel()
        fi = flank_info.get(tid)
        if fi is None:
            return flat[:tx.length]
        return flat[:fi.total_len][fi.k5:fi.k5 + tx.length]

    # Partial ORF emission (3'-truncated). Both orientations of an
    # unstranded transcript are scanned when --unstranded-both-strands is set.
    if args.partial_out is not None and label_store:
        from ..data.gtf_writer import extract_partial_orfs
        n_partial = 0
        partial_lorf_counts: dict[str, int] = {}
        with open(args.partial_out, "w") as fh:
            for tid in sorted(label_store):
                if tid not in transcripts:
                    continue
                tx = transcripts[tid]
                flat_lbl = _tx_labels(tid, tx)
                for orf_start, orf_end in extract_partial_orfs(flat_lbl):
                    if args.min_coding_length > 0 and (orf_end - orf_start) < args.min_coding_length:
                        continue
                    if args.min_utr_5 > 0 and orf_start < args.min_utr_5:
                        continue
                    lc = (classify_lorf(tx_seqs[tid], orf_start)
                          if classify_lorf is not None and tid in tx_seqs else None)
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
                flat_lbl = _tx_labels(tid, tx)
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
        for f in tmp_files:
            try:
                f.unlink()
            except (FileNotFoundError, OSError):
                pass
    return 0
