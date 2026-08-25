"""Project reference CDS onto StringTie-assembled transcripts.

Inputs
------
* StringTie GTF (exons for assembled transcripts)
* Reference GFF/GTF (CDS + stop_codon features for a curated annotation)

For each StringTie transcript the module decides one of:
* ``ir_only``                 - no overlapping reference CDS         -> all-IR negative
* ``kept_single`` / ``kept_multi`` - >=1 complete CDS fully contained -> keep the longest
* ``kept_partial``            - complete ref CDS overlaps same-strand but does
                                 not fully fit on the transcript     -> label the
                                 longest contiguous projectable run as CDS, rest IR
                                 (also recorded in ``ProjectionResult.partial`` so
                                 downstream training can downweight boundary
                                 uncertainty)
* ``antisense_ir``            - only opposite-strand ref CDS overlaps -> all-IR
                                 (transcribed but no on-strand coding signal)
* ``ref_partial_ir``          - only incomplete reference CDS overlaps (frame
                                 not derivable)                       -> all-IR

No StringTie transcript is dropped: every transcript becomes a training example.

Labels per transcript position (int8):
    0 IR   1 START   2 E1   3 E2   4 E0   5 STOP
where ``START`` marks the A of ATG and ``STOP`` marks the last base of the stop
codon. Between them the cycle ``E1, E2, E0`` is repeated, matching the project
convention ``START - E1 - E2 - E0 - ... - E1 - STOP``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
_FRAME_CYCLE = (E1, E2, E0)


@dataclass
class StringTieTranscript:
    tid: str
    contig: str
    strand: str
    exons: list[tuple[int, int]]

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
    cds_parts: list[tuple[int, int]]
    has_stop_codon: bool

    @property
    def total_cds_len(self) -> int:
        return sum(e - s for s, e in self.cds_parts)

    @property
    def g_span(self) -> tuple[int, int]:
        return self.cds_parts[0][0], self.cds_parts[-1][1]

    @property
    def complete(self) -> bool:
        if self.total_cds_len < 6:
            return False
        if self.total_cds_len % 3 != 0:
            return False
        return True


@dataclass
class ProjectionResult:
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    chosen_ref: dict[str, str] = field(default_factory=dict)
    partial: dict[str, dict] = field(default_factory=dict)


def _parse_attr(attr_col: str, key: str) -> str | None:
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
    """Parse a StringTie GTF, collecting exon features per transcript."""
    exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        tid = _parse_attr(f[8], "transcript_id")
        if tid is None:
            continue
        start = int(f[3]) - 1
        end = int(f[4])
        exons[tid].append((start, end))
        meta[tid] = (f[0], f[6])

    out: dict[str, StringTieTranscript] = {}
    for tid, es in exons.items():
        contig, strand = meta[tid]
        out[tid] = StringTieTranscript(
            tid=tid, contig=contig, strand=strand,
            exons=sorted(es),
        )
    return out


def parse_reference_cds(path: Path) -> dict[str, RefTranscript]:
    """Parse a reference GFF/GTF, keeping CDS and stop_codon features per transcript."""
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    stops: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
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

    out: dict[str, RefTranscript] = {}
    for tid, parts in cds.items():
        merged = _merge_intervals(parts + stops.get(tid, []))
        contig, strand = meta[tid]
        out[tid] = RefTranscript(
            tid=tid, contig=contig, strand=strand,
            cds_parts=merged,
            has_stop_codon=bool(stops.get(tid)),
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


def _genomic_to_transcript(t: StringTieTranscript, g: int) -> int | None:
    if t.strand == "+":
        offset = 0
        for s, e in t.exons:
            if s <= g < e:
                return offset + (g - s)
            offset += (e - s)
    else:
        offset = 0
        for s, e in reversed(t.exons):
            if s <= g < e:
                return offset + (e - 1 - g)
            offset += (e - s)
    return None


def _project_ref_onto_transcript(
    t: StringTieTranscript, r: RefTranscript,
) -> tuple[int, int] | None:
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
    bases: list[int] = []
    for cs, ce in r.cds_parts:
        bases.extend(range(cs, ce))
    if r.strand == "-":
        bases.reverse()
    return bases


def _project_partial(
    t: StringTieTranscript, r: RefTranscript,
) -> tuple[int, int, int, int] | None:
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


def build_labels(length: int,
                 orf: tuple[int, int] | None = None) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int8)
    if orf is None:
        return labels
    start, end = orf
    labels[start] = START
    labels[end] = STOP
    for i in range(1, end - start):
        labels[start + i] = _FRAME_CYCLE[(i - 1) % 3]
    return labels


def build_partial_labels(
    length: int,
    t_start: int,
    t_end: int,
    ref_start_pos: int,
    ref_end_pos: int,
    ref_total_len: int,
) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int8)
    for k in range(t_end - t_start + 1):
        ref_pos = ref_start_pos + k
        t_pos = t_start + k
        if ref_pos == 0:
            labels[t_pos] = START
        elif ref_pos == ref_total_len - 1:
            labels[t_pos] = STOP
        else:
            labels[t_pos] = _FRAME_CYCLE[(ref_pos - 1) % 3]
    return labels


def project_labels(
    stringtie_gtf: Path | str,
    reference_gff: Path | str,
) -> ProjectionResult:
    """Run the full labeling pipeline for one species."""
    st = parse_stringtie_gtf(Path(stringtie_gtf))
    refs = parse_reference_cds(Path(reference_gff))
    refs_by_contig: dict[str, list[RefTranscript]] = defaultdict(list)
    for r in refs.values():
        refs_by_contig[r.contig].append(r)

    result = ProjectionResult()

    for tid, t in st.items():
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
            result.labels[tid] = build_labels(t.length)
            result.stats["antisense_ir"] += 1
            continue

        contained = []
        complete_same_strand: list[RefTranscript] = []
        for r in same_strand:
            if not r.complete:
                continue
            complete_same_strand.append(r)
            orf = _project_ref_onto_transcript(t, r)
            if orf is not None:
                contained.append((r, orf))

        if contained:
            contained.sort(key=lambda ro: (-ro[0].total_cds_len, ro[0].tid))
            chosen_r, chosen_orf = contained[0]
            result.labels[tid] = build_labels(t.length, chosen_orf)
            result.chosen_ref[tid] = chosen_r.tid
            result.stats["kept_multi" if len(contained) > 1 else "kept_single"] += 1
            continue

        if not complete_same_strand:
            result.labels[tid] = build_labels(t.length)
            result.stats["ref_partial_ir"] += 1
            continue

        best_partial: tuple[int, RefTranscript, tuple[int, int, int, int]] | None = None
        for r in complete_same_strand:
            proj = _project_partial(t, r)
            if proj is None:
                continue
            run_len = proj[1] - proj[0] + 1
            if best_partial is None or run_len > best_partial[0]:
                best_partial = (run_len, r, proj)

        if best_partial is None:
            result.labels[tid] = build_labels(t.length)
            result.stats["ref_partial_ir"] += 1
            continue

        _, chosen_r, (t_s, t_e, ref_s, ref_e) = best_partial
        result.labels[tid] = build_partial_labels(
            t.length, t_s, t_e, ref_s, ref_e, chosen_r.total_cds_len,
        )
        result.chosen_ref[tid] = chosen_r.tid
        result.partial[tid] = {
            "has_5p": ref_s == 0,
            "has_3p": ref_e == chosen_r.total_cds_len - 1,
            "t_start": t_s,
            "t_end": t_e,
            "ref_start_pos": ref_s,
            "ref_end_pos": ref_e,
            "ref_total_len": chosen_r.total_cds_len,
        }
        result.stats["kept_partial"] += 1

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
            "tid\thas_5p\thas_3p\tt_start\tt_end"
            "\tref_start_pos\tref_end_pos\tref_total_len\n"
        )
        for tid in sorted(result.partial):
            p = result.partial[tid]
            fh.write(
                f"{tid}\t{int(p['has_5p'])}\t{int(p['has_3p'])}"
                f"\t{p['t_start']}\t{p['t_end']}"
                f"\t{p['ref_start_pos']}\t{p['ref_end_pos']}"
                f"\t{p['ref_total_len']}\n"
            )

    return {"labels": npz_path, "stats": stats_path, "partial": partial_path}


def subset_fasta(fasta_in: Path | str, fasta_out: Path | str,
                 keep_ids: set[str]) -> int:
    """Stream ``fasta_in`` and write only records whose id is in ``keep_ids``."""
    fasta_in = Path(fasta_in)
    fasta_out = Path(fasta_out)
    n = 0
    keep = False
    with fasta_in.open("r", encoding="utf-8") as fh_in, \
            fasta_out.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            if line.startswith(">"):
                header_id = line[1:].strip().split()[0]
                keep = header_id in keep_ids
                if keep:
                    fh_out.write(line)
                    n += 1
            elif keep:
                fh_out.write(line)
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Project reference CDS onto "
                                 "StringTie-assembled transcripts.")
    ap.add_argument("--stringtie-gtf", type=Path, required=True)
    ap.add_argument("--reference-gff", type=Path, required=True)
    ap.add_argument("--transcripts-fa", type=Path, required=True,
                    help="Transcripts FASTA from `gffread -w`.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    result = project_labels(args.stringtie_gtf, args.reference_gff)
    paths = write_outputs(result, args.out_dir)

    fasta_out = args.out_dir / "transcripts_labelled.fa"
    n_kept = subset_fasta(args.transcripts_fa, fasta_out,
                          keep_ids=set(result.labels))
    print(f"wrote {n_kept} labelled transcripts -> {fasta_out}")
    print(f"labels -> {paths['labels']}")
    print(f"stats  -> {paths['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
