# `drusilla filter-subseq` — reference

Drop predicted isoforms whose spliced CDS is a (near-)sub-sequence of
another isoform in the same locus. Runs on any prediction GTF with
StringTie-style `transcript_id`s (`STRG.<gene>.<iso>`); the locus is
derived by stripping the trailing `.<iso>`.

Also available as an inline post-step in `drusilla annotate` via
`--subseq-collapse` (see [annotate.md](annotate.md)).

## Definition

For two same-locus, same-strand predictions A and B with CDS block
lists `a_exons` and `b_exons` (sorted ascending by genomic start):

- **`m == 1`** (A single-exon): A's block must lie fully inside one of
  B's blocks.
- **`m >= 2`**: A's leftmost block ends at B's matching block's end
  (its start may be `>=` B's); A's rightmost block starts at B's
  matching block's start (its end may be `<=` B's); A's interior blocks
  match B's exactly. This guarantees A's spliced sequence is a
  contiguous substring of B's on either strand.

Frame is not checked — within a StringTie locus, the model runs on the
same exon nucleotides across isoforms, so shared-exon frames agree in
practice.

## Tolerance knobs

All default to strict (0 / off).

| Flag | Applies to | Meaning |
|---|---|---|
| `--terminal-overhang-nt K` | leftmost/rightmost block only, `m >= 2` | A may extend up to `K` nt beyond the matched keeper block at the terminal side (the shared splice junction on the other side must still match exactly). |
| `--splice-shift-nt S`      | interior blocks, `m >= 2` | Interior boundaries may differ by up to `S` nt on each side, provided each shift is a multiple of 3 (in-frame alternative splice site). |
| `--allow-exon-skip`        | any | A's exons may correspond to a non-contiguous ordered sub-chain of B's exons. Boundary rules per matched pair still apply. |

## Usage

```bash
drusilla filter-subseq \
  --orfs-gtf    results/orfs.gtf \
  --out-gtf     results/orfs.filtered.gtf \
  --report-tsv  results/orfs.dropped_subseq.tsv
```

Relaxed matching, in-frame splice-site drift up to 6 nt:

```bash
drusilla filter-subseq \
  --orfs-gtf    results/orfs.gtf \
  --out-gtf     results/orfs.filtered.gtf \
  --report-tsv  results/orfs.dropped_subseq.tsv \
  --terminal-overhang-nt 15 \
  --splice-shift-nt      6
```

## Outputs

- `--out-gtf` — filtered GTF (dropped isoforms removed; kept isoforms
  emit verbatim CDS lines, so `source` / attribute columns survive
  unchanged).
- `--report-tsv` — three columns: `dropped_tid`, `keeper_tid`,
  `reason`. When several drops chain (A -> B, B -> C), the report is
  re-rooted so both `dropped_tid`s point at a kept `keeper_tid`.

## When to use

- After `drusilla annotate` on multi-isoform StringTie assemblies where
  the model happily emits the ORF on every isoform of a gene, most of
  which just re-spell the same coding sequence with different UTR /
  intron structure.
- As an alternative to `drusilla annotate --single-isoform`, which
  collapses more aggressively (one CDS per StringTie gene, longest
  wins) and is not sensitive to genuine coding-region-differing
  isoforms.

## Complexity

Per StringTie locus with `k` isoforms, worst-case comparisons are
`O(k^2)` times per-pair block matching (`O(m*n)` strict, `O(m*n)` memoised
with `--allow-exon-skip`). In practice a StringTie gene has a handful of
isoforms; the whole step runs in seconds to a couple of minutes on a
typical vertebrate/plant genome.
