"""Transcript-model augmentation for ORF calling on assembled transcripts.

Three mechanisms that matter for compact genomes (fungi) but are harmless
elsewhere:

1. **Unstranded transcripts.** StringTie emits strand ``.`` for most
   single-exon transcripts of an unstranded library and gffread extracts
   them as the forward genomic sequence. :func:`augment_unstranded` keeps
   that forward copy as a ``+`` model and adds a reverse-complement copy
   (``<tid>__rc``, strand ``-``) so minus-strand intronless genes can be
   called at all.
2. **Genomic flanks.** :func:`build_flank` extends a transcript model by
   up to ``k5`` genomic bases upstream and ``k3`` downstream (transcript
   orientation) and returns the flank sequences plus the padded exon
   model. ORFs that run into a flank can then be projected to the genome
   directly ("ORF completion" for 5p/3p-truncated assemblies).
3. **Several ORFs per transcript.** :func:`assign_orf_ids` gives the
   longest ORF the transcript id and later ones ``<tid>.orf2`` etc.;
   :func:`base_tid` strips both suffixes for joins back to the assembly.

Everything here is pure Python so it can be unit-tested without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .label_transcripts import StringTieTranscript

RC_SUFFIX = "__rc"
ORF_SUFFIX = ".orf"

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rev_comp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def sanitize(seq: str) -> str:
    """Upper-case and map anything outside ACGTN to N."""
    seq = seq.upper()
    return "".join(c if c in "ACGTN" else "N" for c in seq)


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------

def is_rc_tid(tid: str) -> bool:
    return base_orf_tid(tid).endswith(RC_SUFFIX)


def base_orf_tid(tid: str) -> str:
    """Strip a ``.orfN`` suffix (keeps ``__rc``)."""
    head, sep, tail = tid.rpartition(ORF_SUFFIX)
    if sep and tail.isdigit():
        return head
    return tid


def base_tid(tid: str) -> str:
    """Strip ``.orfN`` and ``__rc`` suffixes -> assembler transcript id."""
    t = base_orf_tid(tid)
    if t.endswith(RC_SUFFIX):
        t = t[: -len(RC_SUFFIX)]
    return t


def assign_orf_ids(tid: str, n: int) -> list[str]:
    """Ids for ``n`` ORFs on one transcript: ``tid, tid.orf2, tid.orf3, ...``"""
    return [tid] + [f"{tid}{ORF_SUFFIX}{i}" for i in range(2, n + 1)]


# ---------------------------------------------------------------------------
# unstranded transcripts -> both orientations
# ---------------------------------------------------------------------------

def augment_unstranded(
    transcripts: dict[str, StringTieTranscript],
) -> tuple[dict[str, StringTieTranscript], list[str]]:
    """Add a ``-`` reverse-complement model for every unstranded transcript.

    Returns the new mapping (the forward copy keeps its id and is ``+``)
    and the list of base ids that received an ``__rc`` twin.
    """
    out: dict[str, StringTieTranscript] = {}
    rc_tids: list[str] = []
    for tid, tx in transcripts.items():
        if tx.unstranded:
            out[tid] = replace(tx, strand="+")
            out[tid + RC_SUFFIX] = replace(tx, tid=tid + RC_SUFFIX, strand="-")
            rc_tids.append(tid)
        else:
            out[tid] = tx
    return out, rc_tids


def iter_fasta(path: Path | str) -> Iterable[tuple[str, str]]:
    """Yield ``(id, sequence)`` records; id is the header up to whitespace."""
    name: str | None = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                header = line[1:].strip()
                name = header.split()[0] if header else ""
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def write_fasta(path: Path | str, records: Iterable[tuple[str, str]], width: int = 60) -> int:
    n = 0
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")
            if not seq:
                fh.write("\n")
            n += 1
    return n


def rewrite_fasta(
    in_fa: Path | str,
    out_fa: Path | str,
    transform: Callable[[str, str], Iterable[tuple[str, str]]],
) -> int:
    """Stream ``in_fa`` through ``transform(id, seq) -> [(id, seq), ...]``."""
    def gen():
        for name, seq in iter_fasta(in_fa):
            yield from transform(name, seq)
    return write_fasta(out_fa, gen())


def with_rc_copies(rc_tids: set[str]) -> Callable[[str, str], list[tuple[str, str]]]:
    """``rewrite_fasta`` transform adding ``<tid>__rc`` records."""
    def tf(name: str, seq: str) -> list[tuple[str, str]]:
        out = [(name, seq)]
        if name in rc_tids:
            out.append((name + RC_SUFFIX, rev_comp(seq)))
        return out
    return tf


# ---------------------------------------------------------------------------
# genomic flanks
# ---------------------------------------------------------------------------

class GenomeAccess(Protocol):
    def length(self, contig: str) -> int | None: ...
    def fetch(self, contig: str, start: int, end: int) -> str: ...


class DictGenome:
    """``GenomeAccess`` over ``{contig: sequence}`` (tests / small genomes)."""

    def __init__(self, seqs: dict[str, str]):
        self._s = seqs

    def length(self, contig: str) -> int | None:
        s = self._s.get(contig)
        return None if s is None else len(s)

    def fetch(self, contig: str, start: int, end: int) -> str:
        return self._s[contig][start:end]


class FaidxGenome:
    """``GenomeAccess`` over a pyfaidx-indexed FASTA."""

    def __init__(self, fasta_path: Path | str):
        from pyfaidx import Fasta
        self._g = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    def length(self, contig: str) -> int | None:
        if contig not in self._g:
            return None
        return len(self._g[contig])

    def fetch(self, contig: str, start: int, end: int) -> str:
        return str(self._g[contig][start:end])


@dataclass
class FlankInfo:
    """How one transcript was padded before inference.

    Padded FASTA record = ``N*n_pad5 + prefix + tx_seq + suffix + N*n_pad3``
    and always has length ``k5 + tx_len + k3``. ``model`` is the exon model
    of ``prefix + tx_seq + suffix`` (genomic flank bases included), so a
    position ``p`` in the padded record maps to model coordinate
    ``p - n_pad5``.
    """
    k5: int
    k3: int
    prefix: str
    suffix: str
    model: StringTieTranscript
    tx_len: int

    @property
    def a5(self) -> int:
        return len(self.prefix)

    @property
    def a3(self) -> int:
        return len(self.suffix)

    @property
    def n_pad5(self) -> int:
        return self.k5 - self.a5

    @property
    def n_pad3(self) -> int:
        return self.k3 - self.a3

    @property
    def total_len(self) -> int:
        return self.k5 + self.tx_len + self.k3

    def padded_record(self, tx_seq: str) -> str:
        return "N" * self.n_pad5 + self.prefix + tx_seq + self.suffix + "N" * self.n_pad3


def n_pad_only(tx: StringTieTranscript, k5: int) -> FlankInfo:
    """``--prefix-pad-n`` style padding: K literal Ns, no genomic bases."""
    return FlankInfo(k5=k5, k3=0, prefix="", suffix="", model=tx, tx_len=tx.length)


def build_flank(tx: StringTieTranscript, genome: GenomeAccess, k5: int, k3: int) -> FlankInfo:
    """Genomic flanks of up to ``k5`` (5p) and ``k3`` (3p) bases in transcript
    orientation, clamped at contig edges; the padded exon model extends the
    first/last exon accordingly."""
    lo, hi = tx.g_span
    clen = genome.length(tx.contig)
    if clen is None or (k5 == 0 and k3 == 0):
        return FlankInfo(k5=k5, k3=k3, prefix="", suffix="", model=tx, tx_len=tx.length)
    exons = list(tx.exons)
    if tx.strand == "+":
        a5 = min(k5, lo)
        a3 = min(k3, clen - hi)
        prefix = sanitize(genome.fetch(tx.contig, lo - a5, lo)) if a5 else ""
        suffix = sanitize(genome.fetch(tx.contig, hi, hi + a3)) if a3 else ""
        left, right = a5, a3
    else:
        a5 = min(k5, clen - hi)
        a3 = min(k3, lo)
        prefix = rev_comp(sanitize(genome.fetch(tx.contig, hi, hi + a5))) if a5 else ""
        suffix = rev_comp(sanitize(genome.fetch(tx.contig, lo - a3, lo))) if a3 else ""
        left, right = a3, a5
    s0, e0 = exons[0]
    exons[0] = (s0 - left, e0)
    s1, e1 = exons[-1]
    exons[-1] = (s1, e1 + right)
    model = replace(tx, exons=exons)
    return FlankInfo(k5=k5, k3=k3, prefix=prefix, suffix=suffix, model=model, tx_len=tx.length)


# ---------------------------------------------------------------------------
# ORF placement relative to the padded record
# ---------------------------------------------------------------------------

@dataclass
class OrfPlacement:
    intervals: list[tuple[int, int]]        # padded-record coords, half-open
    in_npad: bool                           # touches an N pad (contig edge / --prefix-pad-n)
    ext5: int                               # ORF bases inside the 5p genomic flank
    ext3: int                               # ORF bases inside the 3p genomic flank
    tx_overlap: int                         # coding bases within the transcript proper
    coding_len: int
    tx_start: int                           # ORF start relative to transcript proper (may be < 0)
    tx_end: int                             # ORF end relative to transcript proper (may be > tx_len)

    @property
    def in_flank(self) -> bool:
        return self.ext5 > 0 or self.ext3 > 0

    @property
    def completion(self) -> str | None:
        if self.ext5 and self.ext3:
            return "5p3p"
        if self.ext5:
            return "5p"
        if self.ext3:
            return "3p"
        return None

    def model_intervals(self, fi: FlankInfo) -> list[tuple[int, int]]:
        return [(s - fi.n_pad5, e - fi.n_pad5) for s, e in self.intervals]


def place_orf(cds_intervals: list[tuple[int, int]], fi: FlankInfo) -> OrfPlacement:
    """Locate a decoded ORF (padded-record coords) relative to the N pads,
    the genomic flanks and the assembled transcript proper."""
    iv = sorted(cds_intervals)
    start, end = iv[0][0], iv[-1][1]
    k5, T = fi.k5, fi.tx_len
    tx_lo, tx_hi = k5, k5 + T
    in_npad = start < fi.n_pad5 or end > fi.total_len - fi.n_pad3
    ext5 = sum(max(0, min(e, tx_lo) - s) for s, e in iv)
    ext3 = sum(max(0, e - max(s, tx_hi)) for s, e in iv)
    overlap = sum(max(0, min(e, tx_hi) - max(s, tx_lo)) for s, e in iv)
    coding = sum(e - s for s, e in iv)
    return OrfPlacement(
        intervals=iv, in_npad=in_npad, ext5=ext5, ext3=ext3,
        tx_overlap=overlap, coding_len=coding,
        tx_start=start - k5, tx_end=end - k5,
    )


def clip_5p(cds_intervals: list[tuple[int, int]], k: int) -> list[tuple[int, int]] | None:
    """Drop everything before position ``k`` and advance the new start to the
    next in-frame codon boundary (so the emitted phase is 0). ``None`` if
    nothing coding is left."""
    iv = sorted(cds_intervals)
    s_orig = iv[0][0]
    new: list[tuple[int, int]] = []
    for s, e in iv:
        if e <= k:
            continue
        new.append((max(s, k), e))
    if not new:
        return None
    shift = (3 - (new[0][0] - s_orig) % 3) % 3
    if shift:
        s0, e0 = new[0]
        if s0 + shift >= e0:
            new.pop(0)
        else:
            new[0] = (s0 + shift, e0)
    return new or None


def pick_orientation(fwd_best: int | None, rc_best: int | None) -> str:
    """Which copy of an unstranded transcript to keep: ``'fwd'`` or ``'rc'``.
    Ties and absent rc go to the forward copy."""
    if rc_best is None:
        return "fwd"
    if fwd_best is None:
        return "rc"
    return "rc" if rc_best > fwd_best else "fwd"


# ---------------------------------------------------------------------------
# postprocess policy: what to do with one decoded ORF
# ---------------------------------------------------------------------------

@dataclass
class ResolvedOrf:
    intervals: list[tuple[int, int]]        # coordinates of `model`
    model: StringTieTranscript              # transcript proper or flank-extended model
    completion: str | None                  # None, "5p", "3p", "5p3p"
    tx_start: int                           # ORF start relative to transcript proper
    tx_end: int                             # ORF end relative to transcript proper
    outcome: str                            # counter key


def resolve_orf(
    cds: list[tuple[int, int]],
    tx: StringTieTranscript,
    fi: FlankInfo | None,
    keep: bool = False,
    clip: bool = False,
    min_tx_frac: float = 0.5,
) -> ResolvedOrf | str:
    """Apply the padding policy to one decoded ORF (padded-record coords).

    Returns a :class:`ResolvedOrf`, or a string naming why it was dropped:
    ``dropped_in_npad`` (START/STOP inside literal Ns), ``dropped_flank_only``
    (too little of the ORF inside the assembled transcript), ``dropped_3p_flank``
    (STOP in the 3' flank but ``keep`` is off), ``dropped_in_flank`` (START in
    the 5' flank, neither ``keep`` nor ``clip``, or nothing left after clipping).
    """
    cds = sorted(cds)
    if fi is None:
        return ResolvedOrf(cds, tx, None, cds[0][0], cds[-1][1], "unpadded")
    pl = place_orf(cds, fi)
    if pl.in_npad:
        return "dropped_in_npad"
    if not pl.in_flank:
        iv = [(s - fi.k5, e - fi.k5) for s, e in cds]
        return ResolvedOrf(iv, tx, None, pl.tx_start, pl.tx_end, "unchanged")
    if keep:
        if pl.coding_len == 0 or pl.tx_overlap / pl.coding_len < min_tx_frac:
            return "dropped_flank_only"
        return ResolvedOrf(pl.model_intervals(fi), fi.model, pl.completion,
                           pl.tx_start, pl.tx_end, "completed")
    if pl.ext3 > 0:
        return "dropped_3p_flank"
    if clip:
        clipped = clip_5p(cds, fi.k5)
        if clipped is None:
            return "dropped_in_flank"
        iv = [(s - fi.k5, e - fi.k5) for s, e in clipped]
        return ResolvedOrf(iv, tx, None, iv[0][0], iv[-1][1], "clipped_5p")
    return "dropped_in_flank"
