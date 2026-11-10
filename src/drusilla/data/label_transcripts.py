"""Project reference CDS onto StringTie-assembled transcripts.

Inputs
------
* StringTie GTF (exons for assembled transcripts)
* Reference GFF/GTF (CDS + stop_codon features for a curated annotation)

For each StringTie transcript the module decides one of:
* ``ir_only``                 - no overlapping reference CDS         -> all-IR negative
* ``kept_single``             - one complete CDS fully contained     -> labelled
* ``kept_multi``              - several complete CDS fully contained, all
                                 overlapping each other (isoforms)   -> longest labelled
* ``kept_fused``              - complete CDS of >=2 genes fully contained
                                 without overlapping each other      -> all labelled
                                 (the longest per overlap cluster)
* ``kept_partial``            - complete ref CDS overlaps same-strand but none
                                 fully fits on the transcript        -> label the
                                 longest contiguous projectable run per ref, all
                                 non-overlapping runs, rest IR (also recorded in
                                 ``ProjectionResult.partial``)
* ``antisense_ir``            - only opposite-strand ref CDS overlaps -> all-IR
                                 (transcribed but no on-strand coding signal)
* ``ref_partial_ir``          - only incomplete reference CDS overlaps (frame
                                 not derivable)                       -> all-IR

A transcript with complete genes also receives the non-overlapping visible
runs of truncated further genes (``extra_partial`` counts them); the label
array can hold one frame per position, so overlapping CDS are never labelled
together.

No StringTie transcript is dropped: every transcript becomes a training example.

Strand '.' transcripts: ``parse_stringtie_gtf`` normalises strand '.' to '+'
(gffread extracts such transcripts as the forward genomic sequence) and sets
``StringTieTranscript.unstranded = True``. ``project_labels`` labels both
orientations of them: the forward copy under its own id, compared against '+'
reference CDS, and a reverse-complement copy ``<tid>__rc`` compared against
'-' reference CDS (``rc_copies`` counts them). Whichever orientation holds
the gene is labelled coding and the other one all-IR, which is the pair of
sequences the model is shown at inference. ``write_labelled_fasta`` writes
the reverse-complement sequence for the ``__rc`` records.

Labels per transcript position (int8):
    0 IR   1 START   2 E1   3 E2   4 E0   5 STOP
where ``START`` marks the A of ATG and ``STOP`` marks the last base of the stop
codon. Between them the cycle ``E1, E2, E0`` is repeated, matching the project
convention ``START - E1 - E2 - E0 - ... - E1 - STOP``.

This implementation uses only stdlib + numpy; bricks2marble's ``Anno``/``Transcript``
does not cover StringTie-style exons (its Transcript model stores only CDS parts),
so we parse the two GTFs directly here.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# Label classes.
IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
_FRAME_CYCLE = (E1, E2, E0)  # indexed by (offset_from_start - 1) % 3


@dataclass
class StringTieTranscript:
    """Assembled transcript: id, contig, strand, genomic exons.

    ``strand`` is always '+' or '-'; an assembler strand '.' is stored as '+'
    with ``unstranded=True``.
    """
    tid: str
    contig: str
    strand: str                           # '+' or '-'
    exons: list[tuple[int, int]]          # 0-based half-open; ASC by g_start
    # True when the assembler left the strand undetermined ('.'). gffread
    # extracts such transcripts as the forward genomic sequence, so the
    # model is normalised to '+' and the flag records the ambiguity; the
    # reverse complement is a separate '-' copy (see tx_augment).
    unstranded: bool = False

    @property
    def length(self) -> int:
        return sum(e - s for s, e in self.exons)

    @property
    def g_span(self) -> tuple[int, int]:
        return self.exons[0][0], self.exons[-1][1]


@dataclass
class RefTranscript:
    tid: str
    contig: str
    strand: str
    cds_parts: list[tuple[int, int]]      # 0-based half-open; ASC by g_start
    has_stop_codon: bool                  # whether a stop_codon feature was seen
    # RefSeq GFF3 marks CDS without a verified start or stop with
    # ``partial=true`` (plus ``start_range``/``end_range``) and pseudogene CDS
    # with ``pseudo=true``; neither gives a usable START/STOP, whatever the
    # length modulo 3 says.
    flagged_incomplete: bool = False

    @property
    def total_cds_len(self) -> int:
        return sum(e - s for s, e in self.cds_parts)

    @property
    def g_span(self) -> tuple[int, int]:
        return self.cds_parts[0][0], self.cds_parts[-1][1]

    @property
    def complete(self) -> bool:
        # Complete = multiple of 3 and has a stop codon feature (or the CDS
        # already includes the stop based on a %3 length).  RefSeq typically
        # embeds the stop in CDS; BRAKER splits it out.  Require one of:
        #   * stop_codon feature present (length already accounts for it)
        #   * no stop_codon feature AND CDS length %3 == 0 (CDS includes stop)
        if self.flagged_incomplete:
            return False
        if self.total_cds_len < 6:
            return False
        if self.total_cds_len % 3 != 0:
            return False
        return True


@dataclass
class ProjectionResult:
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # tid -> comma-joined ref tids, in transcript order (one per labelled ORF)
    chosen_ref: dict[str, str] = field(default_factory=dict)
    # tid -> one dict per partially labelled ref, in the order they were
    # added: {ref_tid, has_5p, has_3p, t_start, t_end, ref_start_pos,
    #         ref_end_pos, ref_total_len}
    partial: dict[str, list[dict]] = field(default_factory=dict)


# ---------- parsing ----------

def _parse_attr(attr_col: str, key: str) -> str | None:
    # Supports both GTF (key "value";) and GFF3 (key=value;).
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk and " " not in chunk.split("=", 1)[0]:
            k, v = chunk.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"')
        else:
            parts = chunk.split(None, 1)
            if len(parts) == 2 and parts[0] == key:
                return parts[1].strip().strip('"')
    return None


def parse_stringtie_gtf(path: Path) -> dict[str, StringTieTranscript]:
    """Parse a StringTie GTF, collecting exon features per transcript.

    Strand '.' (or any value other than '+'/'-') is normalised to '+' and
    flagged with ``unstranded=True``, matching gffread's forward-strand
    extraction. Downstream, ``project_labels`` therefore treats these
    transcripts as '+' transcripts.
    """
    exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}     # tid -> (contig, strand)
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        tid = _parse_attr(f[8], "transcript_id")
        if tid is None:
            continue
        start = int(f[3]) - 1       # 1-based incl -> 0-based
        end = int(f[4])             # 1-based incl end -> 0-based exclusive
        exons[tid].append((start, end))
        meta[tid] = (f[0], f[6])

    out: dict[str, StringTieTranscript] = {}
    for tid, es in exons.items():
        contig, strand = meta[tid]
        unstranded = strand not in ("+", "-")
        out[tid] = StringTieTranscript(
            tid=tid, contig=contig, strand="+" if unstranded else strand,
            exons=sorted(es), unstranded=unstranded,
        )
    return out


def parse_reference_cds(path: Path) -> dict[str, RefTranscript]:
    """Parse a reference GFF/GTF, keeping CDS and stop_codon features per transcript."""
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    stops: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    flagged: set[str] = set()
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        feat = f[2]
        if feat not in ("CDS", "stop_codon"):
            continue
        tid = (_parse_attr(f[8], "transcript_id")
               or _parse_attr(f[8], "Parent"))
        if tid is None:
            continue
        start = int(f[3]) - 1
        end = int(f[4])
        (cds if feat == "CDS" else stops)[tid].append((start, end))
        meta[tid] = (f[0], f[6])
        if feat == "CDS" and ("partial=true" in f[8] or "pseudo=true" in f[8]):
            flagged.add(tid)

    out: dict[str, RefTranscript] = {}
    for tid, parts in cds.items():
        # Merge stop_codon intervals into the CDS set if not already covered.
        merged = _merge_intervals(parts + stops.get(tid, []))
        contig, strand = meta[tid]
        out[tid] = RefTranscript(
            tid=tid, contig=contig, strand=strand,
            cds_parts=merged,
            has_stop_codon=bool(stops.get(tid)),
            flagged_incomplete=tid in flagged,
        )
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------- coordinate mapping ----------

def _genomic_to_transcript(t: StringTieTranscript, g: int) -> int | None:
    """Return the transcript coordinate of genomic position ``g`` on ``t``,
    or None if ``g`` does not fall inside any exon of ``t``.
    """
    if t.strand == "+":
        offset = 0
        for s, e in t.exons:                 # ascending
            if s <= g < e:
                return offset + (g - s)
            offset += (e - s)
    else:
        offset = 0
        for s, e in reversed(t.exons):       # 5' -> 3' on the transcript
            if s <= g < e:
                return offset + (e - 1 - g)
            offset += (e - s)
    return None


def _project_ref_onto_transcript(
    t: StringTieTranscript, r: RefTranscript,
) -> tuple[int, int] | None:
    """Map every nucleotide of ``r``'s CDS parts to transcript coordinates in
    ``t``; return ``(orf_start, orf_end)`` inclusive if the CDS is fully
    contained AND contiguous on the transcript, else None.
    """
    t_coords: list[int] = []
    for cs, ce in r.cds_parts:
        for g in range(cs, ce):
            tc = _genomic_to_transcript(t, g)
            if tc is None:
                return None
            t_coords.append(tc)
    t_coords.sort()
    expected = list(range(t_coords[0], t_coords[-1] + 1))
    if t_coords != expected:
        return None
    return t_coords[0], t_coords[-1]


def _ref_bases_in_reading_order(r: RefTranscript) -> list[int]:
    """Return ``r``'s CDS genomic positions in 5'->3' reading order.

    ``cds_parts`` is sorted ascending by genomic start. For '+' strand refs
    that already matches reading order; for '-' strand refs we reverse it.
    """
    bases: list[int] = []
    for cs, ce in r.cds_parts:
        bases.extend(range(cs, ce))
    if r.strand == "-":
        bases.reverse()
    return bases


def _project_partial(
    t: StringTieTranscript, r: RefTranscript,
) -> tuple[int, int, int, int] | None:
    """Find the longest run of consecutive ref CDS positions that map to
    consecutive transcript positions on ``t``.

    Returns ``(t_start, t_end, ref_start_pos, ref_end_pos)`` inclusive on both
    axes, where ``ref_*_pos`` is the 0-based index into the CDS in reading
    order (0 = first base of START codon). Returns ``None`` if no base maps.

    Used for ``kept_partial`` rescue when the full ref CDS is either truncated
    by the assembly's exon boundaries or split by a missing intron.
    """
    ref_g = _ref_bases_in_reading_order(r)
    t_coords = [_genomic_to_transcript(t, g) for g in ref_g]

    best: tuple[int, int, int, int, int] | None = None
    i = 0
    n = len(t_coords)
    while i < n:
        if t_coords[i] is None:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and t_coords[j + 1] is not None
            and t_coords[j + 1] == t_coords[j] + 1
        ):
            j += 1
        length = j - i + 1
        if best is None or length > best[0]:
            best = (length, i, j, t_coords[i], t_coords[j])
        i = j + 1

    if best is None:
        return None
    _, ref_s, ref_e, t_s, t_e = best
    return (t_s, t_e, ref_s, ref_e)


# ---------- labeling ----------

def _write_orf(labels: np.ndarray, start: int, end: int) -> None:
    """Write one complete ORF (inclusive transcript coords) into ``labels``."""
    labels[start] = START
    labels[end] = STOP
    for i in range(1, end - start):
        labels[start + i] = _FRAME_CYCLE[(i - 1) % 3]


def build_labels(
    length: int,
    orf: tuple[int, int] | Iterable[tuple[int, int]] | None = None,
) -> np.ndarray:
    """All-IR labels of ``length`` with one ORF, or several non-overlapping
    ORFs, written in."""
    labels = np.zeros(length, dtype=np.int8)   # IR everywhere
    if orf is None:
        return labels
    orfs = [orf] if isinstance(orf[0], (int, np.integer)) else list(orf)
    for start, end in orfs:
        _write_orf(labels, start, end)
    return labels


def _write_partial(
    labels: np.ndarray,
    t_start: int,
    t_end: int,
    ref_start_pos: int,
    ref_end_pos: int,
    ref_total_len: int,
) -> None:
    """Write the visible run of a partially projected ref CDS into ``labels``.

    Frame labels come from the ref reading position, plus START at
    ``ref_pos==0`` and STOP at ``ref_pos==ref_total_len-1`` if those bases
    happen to be visible.
    """
    for k in range(t_end - t_start + 1):
        ref_pos = ref_start_pos + k
        t_pos = t_start + k
        if ref_pos == 0:
            labels[t_pos] = START
        elif ref_pos == ref_total_len - 1:
            labels[t_pos] = STOP
        else:
            labels[t_pos] = _FRAME_CYCLE[(ref_pos - 1) % 3]


def build_partial_labels(
    length: int,
    t_start: int,
    t_end: int,
    ref_start_pos: int,
    ref_end_pos: int,
    ref_total_len: int,
) -> np.ndarray:
    """Label only the visible portion of a partially projected ref CDS;
    everything outside the visible run is IR."""
    labels = np.zeros(length, dtype=np.int8)
    _write_partial(labels, t_start, t_end, ref_start_pos, ref_end_pos, ref_total_len)
    return labels


def _cluster_by_overlap(
    contained: list[tuple[RefTranscript, tuple[int, int]]],
) -> list[list[tuple[RefTranscript, tuple[int, int]]]]:
    """Group contained CDS whose transcript intervals overlap (isoforms of one
    gene); clusters come back in transcript order."""
    clusters: list[list[tuple[RefTranscript, tuple[int, int]]]] = []
    end = -1
    for r, orf in sorted(contained, key=lambda ro: (ro[1][0], ro[1][1], ro[0].tid)):
        if clusters and orf[0] <= end:
            clusters[-1].append((r, orf))
            end = max(end, orf[1])
        else:
            clusters.append([(r, orf)])
            end = orf[1]
    return clusters


def _overlaps_any(iv: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
    return any(iv[0] <= e and s <= iv[1] for s, e in covered)


# ---------- main projection logic ----------

def project_labels(
    stringtie_gtf: Path | str,
    reference_gff: Path | str,
    both_orientations: bool = True,
) -> ProjectionResult:
    """Run the full labeling pipeline for one species.

    Returns per-transcript labels and a category-count dict.

    With ``both_orientations`` every strand '.' transcript is labelled twice:
    as '+' under its own id and as '-' under ``<tid>__rc`` (the reverse
    complement; see ``tx_augment.augment_unstranded``). Complete reference
    CDS fully contained in a transcript are clustered by overlap, the longest
    per cluster is labelled, and visible runs of truncated further genes are
    added where they do not overlap a labelled region.
    """
    st = parse_stringtie_gtf(Path(stringtie_gtf))
    refs = parse_reference_cds(Path(reference_gff))
    result = ProjectionResult()

    if both_orientations:
        # tx_augment imports StringTieTranscript from here: import lazily.
        from .tx_augment import augment_unstranded
        st, rc_tids = augment_unstranded(st)
        if rc_tids:
            result.stats["rc_copies"] = len(rc_tids)

    # Index reference transcripts by contig for quick lookup.
    refs_by_contig: dict[str, list[RefTranscript]] = defaultdict(list)
    for r in refs.values():
        refs_by_contig[r.contig].append(r)

    for tid, t in st.items():
        # Candidate refs on same contig with genomic overlap.
        t_s, t_e = t.g_span
        overlapping = [
            r for r in refs_by_contig.get(t.contig, [])
            if _ranges_overlap(r.g_span, (t_s, t_e))
        ]
        if not overlapping:
            result.labels[tid] = build_labels(t.length)
            result.stats["ir_only"] += 1
            continue

        same_strand = [r for r in overlapping if r.strand == t.strand]
        if not same_strand:
            # Antisense rescue: only opposite-strand CDSes overlap. There is
            # no on-strand coding signal so emit all-IR labels; this is the
            # antisense-lncRNA signal we want the model to learn.
            result.labels[tid] = build_labels(t.length)
            result.stats["antisense_ir"] += 1
            continue

        complete_same_strand = [r for r in same_strand if r.complete]
        if not complete_same_strand:
            # Only incomplete refs overlap; reading frame is not derivable so
            # we emit all-IR rather than fabricate frame labels.
            result.labels[tid] = build_labels(t.length)
            result.stats["ref_partial_ir"] += 1
            continue

        contained: list[tuple[RefTranscript, tuple[int, int]]] = []
        not_contained: list[RefTranscript] = []
        for r in complete_same_strand:
            orf = _project_ref_onto_transcript(t, r)
            if orf is None:
                not_contained.append(r)
            else:
                contained.append((r, orf))

        labels = build_labels(t.length)
        covered: list[tuple[int, int]] = []
        chosen: list[str] = []

        # Complete genes: one ORF per overlap cluster, the longest CDS of the
        # cluster (tie-break by TID). Several clusters = a fused transcript.
        clusters = _cluster_by_overlap(contained)
        for cluster in clusters:
            cluster.sort(key=lambda ro: (-ro[0].total_cds_len, ro[0].tid))
            r, orf = cluster[0]
            _write_orf(labels, *orf)
            covered.append(orf)
            chosen.append(r.tid)

        # Partial rescue: complete refs that overlap but do not fit (5'/3'
        # assembly truncation or missing intron). Longest visible run first;
        # a run is labelled only where nothing else is labelled yet.
        runs: list[tuple[int, RefTranscript, tuple[int, int, int, int]]] = []
        for r in not_contained:
            proj = _project_partial(t, r)
            if proj is not None:
                runs.append((proj[1] - proj[0] + 1, r, proj))
        runs.sort(key=lambda x: (-x[0], x[1].tid))
        n_partial = 0
        for _, r, (p_s, p_e, ref_s, ref_e) in runs:
            if _overlaps_any((p_s, p_e), covered):
                continue
            _write_partial(labels, p_s, p_e, ref_s, ref_e, r.total_cds_len)
            covered.append((p_s, p_e))
            chosen.append(r.tid)
            n_partial += 1
            result.partial.setdefault(tid, []).append({
                "ref_tid": r.tid,
                "has_5p": ref_s == 0,
                "has_3p": ref_e == r.total_cds_len - 1,
                "t_start": p_s,
                "t_end": p_e,
                "ref_start_pos": ref_s,
                "ref_end_pos": ref_e,
                "ref_total_len": r.total_cds_len,
            })

        if not chosen:
            # g_span overlapped but no base projected (shouldn't normally
            # happen). Fall back to all-IR rather than drop.
            result.labels[tid] = labels
            result.stats["ref_partial_ir"] += 1
            continue

        result.labels[tid] = labels
        result.chosen_ref[tid] = ",".join(chosen)
        if len(clusters) >= 2:
            result.stats["kept_fused"] += 1
        elif len(clusters) == 1:
            result.stats["kept_multi" if len(clusters[0]) > 1 else "kept_single"] += 1
        else:
            result.stats["kept_partial"] += 1
        if clusters and n_partial:
            result.stats["extra_partial"] += n_partial

    return result


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def write_outputs(
    result: ProjectionResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "labels.npz"
    np.savez_compressed(npz_path, **result.labels)

    stats_path = out_dir / "stats.tsv"
    with stats_path.open("w", encoding="utf-8") as fh:
        fh.write("category\tcount\n")
        for k in sorted(result.stats):
            fh.write(f"{k}\t{result.stats[k]}\n")

    partial_path = out_dir / "partial.tsv"
    with partial_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "tid\tref_tid\thas_5p\thas_3p\tt_start\tt_end"
            "\tref_start_pos\tref_end_pos\tref_total_len\n"
        )
        for tid in sorted(result.partial):
            for p in result.partial[tid]:
                fh.write(
                    f"{tid}\t{p['ref_tid']}\t{int(p['has_5p'])}\t{int(p['has_3p'])}"
                    f"\t{p['t_start']}\t{p['t_end']}"
                    f"\t{p['ref_start_pos']}\t{p['ref_end_pos']}"
                    f"\t{p['ref_total_len']}\n"
                )

    return {"labels": npz_path, "stats": stats_path, "partial": partial_path}


def _iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(id, sequence)`` from a plain or gzipped FASTA; id is the
    header up to the first whitespace."""
    opener = gzip.open if path.suffix == ".gz" else open
    name: str | None = None
    buf: list[str] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name = line[1:].strip().split()[0] if line[1:].strip() else ""
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def write_labelled_fasta(
    fasta_in: Path | str,
    fasta_out: Path | str,
    keep_ids: set[str],
    width: int = 60,
) -> int:
    """Write the transcripts in ``keep_ids`` from ``fasta_in`` (plain or .gz)
    to ``fasta_out``; for every ``<id>__rc`` in ``keep_ids`` also write the
    reverse complement of ``<id>``. Returns the number of records written.
    """
    from .tx_augment import RC_SUFFIX, rev_comp

    n = 0
    with Path(fasta_out).open("w", encoding="utf-8") as fh:
        for name, seq in _iter_fasta(Path(fasta_in)):
            records = []
            if name in keep_ids:
                records.append((name, seq))
            if name + RC_SUFFIX in keep_ids:
                records.append((name + RC_SUFFIX, rev_comp(seq)))
            for rec_name, rec_seq in records:
                fh.write(f">{rec_name}\n")
                for i in range(0, len(rec_seq), width):
                    fh.write(rec_seq[i:i + width] + "\n")
                n += 1
    return n


def subset_fasta(fasta_in: Path | str, fasta_out: Path | str,
                 keep_ids: set[str]) -> int:
    """Stream ``fasta_in`` and write only records whose id is in ``keep_ids``
    (no reverse-complement copies). Kept for callers of the old name."""
    return write_labelled_fasta(fasta_in, fasta_out,
                                {k for k in keep_ids if not k.endswith("__rc")})


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Project reference CDS onto "
                                 "StringTie-assembled transcripts.")
    ap.add_argument("--stringtie-gtf", type=Path, required=True)
    ap.add_argument("--reference-gff", type=Path, required=True)
    ap.add_argument("--transcripts-fa", type=Path, required=True,
                    help="Transcripts FASTA from `gffread -w` (plain or .gz).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--single-orientation", action="store_true",
                    help="Label strand '.' transcripts as '+' only, without "
                         "the reverse-complement '<tid>__rc' copy.")
    args = ap.parse_args(argv)

    result = project_labels(args.stringtie_gtf, args.reference_gff,
                            both_orientations=not args.single_orientation)
    paths = write_outputs(result, args.out_dir)

    fasta_out = args.out_dir / "transcripts_labelled.fa"
    n_kept = write_labelled_fasta(args.transcripts_fa, fasta_out,
                                  keep_ids=set(result.labels))
    if n_kept != len(result.labels):
        print(f"warning: {len(result.labels) - n_kept} labelled transcripts "
              f"are missing from {args.transcripts_fa}")
    print(f"wrote {n_kept} labelled transcripts -> {fasta_out}")
    print(f"labels -> {paths['labels']}")
    print(f"stats  -> {paths['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
