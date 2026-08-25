"""Project transcript-coordinate CDS intervals onto genomic coordinates."""

from __future__ import annotations


def project_tx_intervals_to_genomic(
    tx_id: str,
    cds_intervals: list[tuple[int, int]],
    tx,
    source: str,
    gene_id: str | None = None,
    lorf_class: str | None = None,
) -> list[str]:
    """Project half-open transcript-coord CDS intervals onto the genome.

    ``tx`` must expose ``.contig`` (str), ``.strand`` ('+' or '-'), and
    ``.exons`` (list of (g_start, g_end) 0-based half-open, ASC by
    g_start) - i.e. a ``StringTieTranscript`` from
    ``drusilla.data.label_transcripts``.

    Returns one CDS GTF line per genomic sub-interval, sorted ascending
    by genomic start.
    """
    gid = gene_id if gene_id is not None else tx_id
    out: list[str] = []
    for orf_tx_start, orf_tx_end in cds_intervals:
        if orf_tx_start >= orf_tx_end:
            continue
        exons_in_tx_order = (
            list(tx.exons) if tx.strand == "+" else list(reversed(tx.exons))
        )
        cumulative = 0
        per_segment: list[tuple[int, int, int]] = []
        for g_start, g_end in exons_in_tx_order:
            exon_len = g_end - g_start
            lo = max(orf_tx_start, cumulative)
            hi = min(orf_tx_end, cumulative + exon_len)
            if lo < hi:
                off_lo = lo - cumulative
                off_hi = hi - cumulative
                if tx.strand == "+":
                    g_lo, g_hi = g_start + off_lo, g_start + off_hi
                else:
                    g_lo, g_hi = g_end - off_hi, g_end - off_lo
                per_segment.append((g_lo, g_hi, lo))
            cumulative += exon_len
            if cumulative >= orf_tx_end:
                break
        per_segment.sort()
        lorf_attr = f' lorf_class "{lorf_class}";' if lorf_class is not None else ""
        for g_lo, g_hi, tx_pos in per_segment:
            phase = (3 - (tx_pos - orf_tx_start) % 3) % 3
            out.append("\t".join([
                tx.contig, source, "CDS",
                str(g_lo + 1), str(g_hi),
                ".", tx.strand, str(phase),
                f'transcript_id "{tx_id}"; gene_id "{gid}";{lorf_attr}',
            ]))
    return out
