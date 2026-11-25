"""Frame recoding: genomes of a nuclear genetic code other than table 1.

The network was trained on table-1 genomes and labels a CDS with an in-frame
TAA, TAG or TGA as non-coding. In a genome whose code reads one or two of
these codons as an amino acid (table 6: TAA and TAG are Gln), nearly every
CDS has such codons. Without retraining, each transcript record is given to
the network three times, once per reading frame, with the reassigned codons
of that frame replaced by a table-1 codon of the same amino acid (table 6:
CAA and CAG). In the copy of the true frame the CDS then reads as a table-1
CDS that ends at a stop codon of the genome's table. An ORF is kept only
when it lies in the frame its copy was recoded for; the replacement changes
no length, so all coordinates stay those of the transcript.

Measured on Tetrahymena thermophila (table 6), exact CDS match to the RefSeq
annotation on a StringTie assembly of 35,375 transcripts: precision 0.51 and
sensitivity 0.48, against 0.44 and 0.43 of the longest ATG-to-TGA ORF;
without recoding the network finds 31 of 26,996 genes.
"""

from __future__ import annotations

from typing import Callable

FRAME_SUFFIX = "__f"

_TABLE1_STOPS = ("TAA", "TAG", "TGA")

# NCBI translation table of a eukaryotic nuclear genome -> {stop codon of
# table 1 that is a sense codon in this table: table-1 codon of the same
# amino acid}. Not listed: tables with a context-dependent stop codon (27,
# 28, 31), tables that differ in a sense codon (12, 26), and the codes of
# organelles and bacteria.
RECODING_BY_TABLE: dict[int, dict[str, str]] = {
    1: {},
    6: {"TAA": "CAA", "TAG": "CAG"},      # Gln
    10: {"TGA": "TGT"},                   # Cys
    15: {"TAG": "CAG"},                   # Gln
    29: {"TAA": "TAT", "TAG": "TAC"},     # Tyr
    30: {"TAA": "GAA", "TAG": "GAG"},     # Glu
}


def recoding_for_table(table: int) -> dict[str, str]:
    try:
        return RECODING_BY_TABLE[int(table)]
    except KeyError:
        raise ValueError(
            f"translation table {table} is not supported "
            f"(supported: {sorted(RECODING_BY_TABLE)})"
        ) from None


def stop_codons_for_table(table: int) -> frozenset[str]:
    """Stop codons of an NCBI translation table."""
    recoding = recoding_for_table(table)
    return frozenset(c for c in _TABLE1_STOPS if c not in recoding)


def recode_frame(seq: str, frame: int, recoding: dict[str, str]) -> str:
    """``seq`` with the reassigned codons of reading frame ``frame`` (0, 1 or
    2: the offset of the first full codon) replaced. Case is not kept for a
    replaced codon; the length does not change."""
    out = [seq[:frame]]
    for i in range(frame, len(seq), 3):
        codon = seq[i:i + 3]
        out.append(recoding.get(codon.upper(), codon))
    return "".join(out)


def frame_copies(recoding: dict[str, str]) -> Callable[[str, str], list[tuple[str, str]]]:
    """``rewrite_fasta`` transform: three records ``<id>__f0|1|2`` per record."""
    def tf(name: str, seq: str) -> list[tuple[str, str]]:
        return [(f"{name}{FRAME_SUFFIX}{f}", recode_frame(seq, f, recoding)) for f in range(3)]
    return tf


def split_frame_id(name: str) -> tuple[str, int]:
    """``<id>__f<frame>`` -> (id, frame)."""
    head, sep, tail = name.rpartition(FRAME_SUFFIX)
    if not sep or tail not in ("0", "1", "2"):
        raise ValueError(f"not a frame copy id: {name}")
    return head, int(tail)
