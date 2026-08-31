"""Drop predicted isoforms whose CDS is a (near-)sub-sequence of another isoform.

A predicted CDS A is a "sub-sequence" of another predicted CDS B (same
locus, same strand) when A's spliced nucleotide CDS appears contiguously
inside B's spliced nucleotide CDS. Operationally (strict defaults):

* m == 1 (A is single-exon)
      A's single CDS block must lie fully inside one of B's CDS blocks.
* m >= 2
      A's interior blocks match B's exactly. A's leftmost (in genome) block
      ends at B's matching block's end (its start may be >= B's). A's
      rightmost block starts at B's matching block's start (its end may be
      <= B's). Together these guarantee A's spliced sequence is a
      contiguous substring of B's, on either strand.

Three tolerance knobs relax the strict rules for near-subsequence cases
(all defaults = 0 / off = strict behavior):

* ``terminal_overhang_nt`` (m >= 2)
      A's leftmost block may extend up to K nt *earlier* than the matched
      B block (start), and A's rightmost block may extend up to K nt
      *later* than the matched B block (end). The shared internal splice
      junction on the other side must still match exactly.
* ``splice_shift_nt`` (m >= 2, interior blocks)
      Interior block boundaries may differ from the matched B block's
      boundaries by up to S nt at each side, provided each shift is a
      multiple of 3 (in-frame alternative splice site).
* ``allow_exon_skip``
      A's exons may correspond to a non-contiguous sub-chain of B's exons.

Frame is not checked. Within one StringTie locus the model runs on the
same exon nucleotides across isoforms, so frame at shared exons is the
same in practice.

Locus grouping: derived from the GTF ``transcript_id`` by stripping the
trailing ``.<iso>`` (StringTie convention ``STRG.<gene>.<iso>``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


_GENE_ID_FROM_TX = re.compile(r"^(.*)\.\d+$")


def _parse_attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def gene_id_from_tx(tid: str) -> str:
    m = _GENE_ID_FROM_TX.match(tid)
    return m.group(1) if m else tid


def parse_prediction_gtf(path: Path) -> dict[str, dict]:
    """Return ``{tid: {contig, strand, exons: [(s, e), ...], lines: [str]}}``.

    ``exons`` are 0-based half-open and sorted ascending by start.
    ``lines`` preserves original GTF lines so the output keeps every
    column verbatim.
    """
    by_tid: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            tid = _parse_attr(f[8], "transcript_id")
            if tid is None:
                continue
            s = int(f[3]) - 1
            e = int(f[4])
            rec = by_tid.setdefault(tid, {
                "contig": f[0],
                "strand": f[6],
                "exons": [],
                "lines": [],
            })
            rec["exons"].append((s, e))
            rec["lines"].append(line)
    for rec in by_tid.values():
        rec["exons"].sort()
    return by_tid


def _match_first(a: tuple[int, int], b: tuple[int, int], overhang: int) -> bool:
    """Leftmost-block rule: shared splice donor (a_e == b_e), a_s may
    extend up to ``overhang`` nt earlier than b_s."""
    return a[1] == b[1] and a[0] >= b[0] - overhang


def _match_last(a: tuple[int, int], b: tuple[int, int], overhang: int) -> bool:
    """Rightmost-block rule: shared splice acceptor (a_s == b_s), a_e may
    extend up to ``overhang`` nt later than b_e."""
    return a[0] == b[0] and a[1] <= b[1] + overhang


def _match_interior(a: tuple[int, int], b: tuple[int, int], shift: int) -> bool:
    """Interior-block rule: each boundary may differ by up to ``shift`` nt,
    with the shift a multiple of 3 (frame-preserving)."""
    ds = a[0] - b[0]
    de = a[1] - b[1]
    return abs(ds) <= shift and abs(de) <= shift and ds % 3 == 0 and de % 3 == 0


def is_subsequence(
    a_exons: list[tuple[int, int]],
    b_exons: list[tuple[int, int]],
    terminal_overhang_nt: int = 0,
    splice_shift_nt: int = 0,
    allow_exon_skip: bool = False,
) -> bool:
    """Is A's spliced CDS a (near-)sub-sequence of B's spliced CDS?

    Both inputs are sorted ascending lists of (start, end) half-open
    genomic intervals. Same contig and strand must already be guaranteed
    by the caller.
    """
    m, n = len(a_exons), len(b_exons)
    if m == 0 or m > n:
        return False
    if m == 1:
        for b_s, b_e in b_exons:
            a_s, a_e = a_exons[0]
            if b_s <= a_s and a_e <= b_e:
                return True
        return False

    if not allow_exon_skip:
        for k in range(n - m + 1):
            b_slice = b_exons[k:k + m]
            if not _match_first(a_exons[0], b_slice[0], terminal_overhang_nt):
                continue
            if not _match_last(a_exons[-1], b_slice[-1], terminal_overhang_nt):
                continue
            if all(_match_interior(a_exons[i], b_slice[i], splice_shift_nt)
                   for i in range(1, m - 1)):
                return True
        return False

    # Exon-skip mode: A's exons form an ordered subset of B's exons.
    memo: dict[tuple[int, int], bool] = {}

    def try_match(i_a: int, i_b: int) -> bool:
        if i_a == m:
            return True
        if i_b >= n:
            return False
        if (i_a, i_b) in memo:
            return memo[(i_a, i_b)]
        is_first = i_a == 0
        is_last = i_a == m - 1
        if is_first:
            block_ok = _match_first(a_exons[0], b_exons[i_b],
                                    terminal_overhang_nt)
        elif is_last:
            block_ok = _match_last(a_exons[-1], b_exons[i_b],
                                   terminal_overhang_nt)
        else:
            block_ok = _match_interior(a_exons[i_a], b_exons[i_b],
                                       splice_shift_nt)
        result = (block_ok and try_match(i_a + 1, i_b + 1)) \
            or try_match(i_a, i_b + 1)
        memo[(i_a, i_b)] = result
        return result

    return try_match(0, 0)


def find_dropable(
    by_tid: dict[str, dict],
    terminal_overhang_nt: int = 0,
    splice_shift_nt: int = 0,
    allow_exon_skip: bool = False,
) -> dict[str, str]:
    """Return ``{dropped_tid: keeper_tid}``.

    Within each (contig, strand, locus) group, drop A if some other
    isoform B in the group has A as a sub-sequence. When two isoforms
    have identical CDS chains, the one with more exons / longer span
    sorts first and is kept; the duplicate is dropped.
    """
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for tid, rec in by_tid.items():
        gid = gene_id_from_tx(tid)
        groups[(rec["contig"], rec["strand"], gid)].append(tid)

    drop_map: dict[str, str] = {}
    for tids in groups.values():
        if len(tids) < 2:
            continue
        tids = sorted(
            tids,
            key=lambda t: (
                -len(by_tid[t]["exons"]),
                -(by_tid[t]["exons"][-1][1] - by_tid[t]["exons"][0][0]),
                t,
            ),
        )
        for i, a_tid in enumerate(tids):
            if a_tid in drop_map:
                continue
            for j in range(len(tids)):
                if i == j:
                    continue
                b_tid = tids[j]
                if b_tid in drop_map:
                    continue
                if is_subsequence(by_tid[a_tid]["exons"],
                                  by_tid[b_tid]["exons"],
                                  terminal_overhang_nt=terminal_overhang_nt,
                                  splice_shift_nt=splice_shift_nt,
                                  allow_exon_skip=allow_exon_skip):
                    drop_map[a_tid] = b_tid
                    break

    # Re-root: if a keeper is itself dropped, chase the chain so the
    # report points at a kept tx.
    for tid in list(drop_map):
        keeper = drop_map[tid]
        seen = {tid}
        while keeper in drop_map and keeper not in seen:
            seen.add(keeper)
            keeper = drop_map[keeper]
        drop_map[tid] = keeper
    return drop_map


def write_filtered(
    by_tid: dict[str, dict],
    drop_map: dict[str, str],
    out_gtf: Path,
    report_tsv: Path | None,
) -> tuple[int, int, int, int]:
    """Write the filtered GTF and (optional) drop-report TSV.

    Returns ``(n_kept_tx, n_kept_cds, n_dropped_tx, n_dropped_cds)``.
    """
    out_gtf = Path(out_gtf)
    out_gtf.parent.mkdir(parents=True, exist_ok=True)

    n_kept_tx = n_kept_cds = n_drop_cds = 0
    with out_gtf.open("w", encoding="utf-8") as fh:
        for tid, rec in by_tid.items():
            if tid in drop_map:
                n_drop_cds += len(rec["lines"])
                continue
            n_kept_tx += 1
            for ln in rec["lines"]:
                fh.write(ln + "\n")
                n_kept_cds += 1

    if report_tsv is not None:
        report_tsv = Path(report_tsv)
        report_tsv.parent.mkdir(parents=True, exist_ok=True)
        with report_tsv.open("w", encoding="utf-8") as fh:
            fh.write("dropped_tid\tkeeper_tid\treason\n")
            for dropped, keeper in sorted(drop_map.items()):
                fh.write(f"{dropped}\t{keeper}\tsub_sequence_of_keeper\n")

    return n_kept_tx, n_kept_cds, len(drop_map), n_drop_cds


def run_filter(
    orfs_gtf: Path,
    out_gtf: Path,
    report_tsv: Path | None = None,
    terminal_overhang_nt: int = 0,
    splice_shift_nt: int = 0,
    allow_exon_skip: bool = False,
) -> dict:
    """End-to-end helper: parse -> find dropable -> write. Returns a stats dict."""
    if splice_shift_nt < 0 or terminal_overhang_nt < 0:
        raise ValueError("tolerance values must be >= 0")

    by_tid = parse_prediction_gtf(Path(orfs_gtf))
    drop_map = find_dropable(
        by_tid,
        terminal_overhang_nt=terminal_overhang_nt,
        splice_shift_nt=splice_shift_nt,
        allow_exon_skip=allow_exon_skip,
    )
    n_kept_tx, n_kept_cds, n_drop_tx, n_drop_cds = write_filtered(
        by_tid, drop_map, Path(out_gtf), report_tsv,
    )
    n_in = len(by_tid)
    return {
        "n_input_tx": n_in,
        "n_dropped_tx": n_drop_tx,
        "n_kept_tx": n_kept_tx,
        "n_kept_cds": n_kept_cds,
        "n_dropped_cds": n_drop_cds,
        "pct_dropped": (100.0 * n_drop_tx / max(1, n_in)),
    }
