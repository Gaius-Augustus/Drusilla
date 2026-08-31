# `drusilla annotate` — reference

End-to-end ORF annotation from a StringTie GTF or a sorted RNA-seq BAM.

## Input modes

Exactly one is required:

| Flag | Meaning |
|---|---|
| `--stringtie-gtf FILE` | Skip StringTie; use this pre-computed GTF. |
| `--bam FILE`           | Run StringTie internally on this sorted BAM. |

Both modes also require `--genome FILE` (the genome FASTA whose contig
names match the GTF/BAM).

If you already have a spliced transcript FASTA (e.g. from a previous
`gffread` run), you can pass `--transcripts-fa FILE` to skip that step
too. The FASTA record IDs must equal the StringTie `transcript_id`s.

## Model selection

Exactly one path chooses the model:

| Flag | Meaning |
|---|---|
| _(default)_             | Uses `--model vertebrates`. |
| `--model NAME`          | Registered model (see `drusilla models list`). Weights auto-downloaded on first use. |
| `--weights FILE`        | Local `.weights.h5`. Requires `--config FILE` to describe the architecture. |

The `--config` flag is auto-selected from the registry when `--model` is
used. Only override it if you know what you're doing.

## Output

Written under `--out-dir` in the default mode:

| File | Description |
|---|---|
| `orfs.gtf`       | **PRIMARY.** Genomic-coordinate GTF, one `CDS` line per genomic sub-interval of every predicted ORF (1-based inclusive; `source = "drusilla"`; shared `transcript_id` / `gene_id` per ORF). Subsequence-isoform collapse is applied by default (see below). |
| `orfs_local.gtf` | Same predictions in transcript (local) coordinates: one `CDS` line per predicted ORF, contig column holds the StringTie `transcript_id`, strand = `+`, phase = 0 at the ORF start. Always in sync with `orfs.gtf` (same set of kept transcripts). |
| `orfs.log`       | b2m annotation log. |

Extra outputs, only produced when their flag is set:

| File | Enabled by |
|---|---|
| `orfs.dropped_subseq.tsv` | `--subseq-report` |
| `<partial-out.gtf>`       | `--partial-out FILE`  (3'-truncated ORFs) |
| `<partial5-out.gtf>`      | `--partial5-out FILE` (5'-truncated ORFs) |
| `stringtie.filtered.gtf`  | Any of `--min-cov / --min-tpm / --drop-unstranded / --drop-single-exon`. |
| `stringtie.gtf`, `transcripts.fa` | Intermediates produced when starting from `--bam` or when gffread ran. Deleted unless `--keep-tmp` is set. |

## Flags

### Inference

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size N`         | auto | NN inference batch size. If omitted, auto-selected from the GPU memory (queried via `nvidia-smi`) and the model's `chunk_len`. Empirical anchors: 40 GB A100 -> ~180, 80 GB A100 -> ~360, smaller GPUs scale down. If `nvidia-smi` is unavailable the fallback is 8. Override any time with an explicit value. |
| `--parallel N`           | 100 | HMM parallel-scan factor (chunks padded to a multiple of N). |
| `--no-codon-emitter`     | off | Drop the hard ATG-at-START / not-stop-at-E2 / stop-at-STOP codon constraints (debug). |
| `--restrict-start-to-ir-start` | off | Force the HMM's initial state distribution to `{IR, START}` (whole-transcript decode should never start mid-codon). |
| `--ir-prior FLOAT`       | 0.5 | Under `--restrict-start-to-ir-start`, `P(start in IR)`. |
| `--threads N`            | 4   | Threads for stringtie / gffread. |
| `--longread`             | off | Pass `-L` to StringTie (PacBio Iso-Seq / ONT). |
| `--keep-tmp`             | off | Do not delete intermediate GTF / FASTA files. |

### 5' context padding

Adds fake upstream context so the model has a left window at position 0.

| Flag | Meaning |
|---|---|
| `--prefix-pad-n K`       | Prepend K literal Ns per transcript. |
| `--flank-bp K`           | Prepend K bp of real genomic context (RC for `-` strand); short chromosome edges are N-padded. Mutually exclusive with `--prefix-pad-n`. |
| `--flank-clip`           | With `--flank-bp`, clip ORFs whose START falls in the flank to the next in-frame codon inside the transcript. Without this, such ORFs are dropped. |

### Reference-free StringTie pre-filters

Approximate the "clean-transcript" curation the training pipeline does
with a reference annotation.

| Flag | Meaning |
|---|---|
| `--min-cov FLOAT`        | Drop StringTie transcripts with `cov` below this. |
| `--min-tpm FLOAT`        | Drop StringTie transcripts with `TPM` below this. |
| `--drop-unstranded`      | Drop transcripts with strand `.`. |
| `--drop-single-exon`     | Drop single-exon transcripts (often intronic noise). |

Typical short-read defaults: `--min-cov 3 --min-tpm 1 --drop-unstranded`.

### Post-decode ORF filters

| Flag | Default | Meaning |
|---|---|---|
| `--min-coding-length N`  | 200 | Drop ORFs whose total CDS length is below N nt. Set to 0 to disable. |
| `--single-isoform`       | off | Keep at most one transcript per StringTie `gene_id` — the one with the longest predicted CDS (ties broken by `transcript_id`). |
| `--min-utr-5 N`          | 0   | Drop ORFs whose START is within N nt of the transcript 5' end (proxy for 5' assembly truncation). |
| `--min-utr-3 N`          | 0   | Drop ORFs whose STOP is within N nt of the transcript 3' end. |

### Optional partial-ORF output

| Flag | Meaning |
|---|---|
| `--partial-out FILE`     | Also write 3'-truncated ORFs (`START ... E*`, no STOP within the transcript). |
| `--partial5-out FILE`    | Also write 5'-truncated ORFs (`E*` at position 0, ending at a STOP; no upstream ATG). |
| `--lorf-class`           | Add a `lorf_class` GTF attribute per ORF (`LORF_UPSTOP` / `sORF_UPSTOP` / `upLORF` / `LORF_NOUPSTOP`). Requires loading transcript sequences into memory. |

### Subsequence-isoform collapse (default ON)

Drops predicted isoforms whose spliced CDS is a (near-)sub-sequence of
another isoform in the same StringTie locus (locus = `transcript_id`
minus the trailing `.<iso>`). Applied automatically to both `orfs.gtf`
and `orfs_local.gtf` so the two files always agree. The same collapse
is also available as a standalone command — see
[filter_subseq.md](filter_subseq.md).

| Flag | Default | Meaning |
|---|---|---|
| `--no-subseq-collapse`            | off (collapse on) | Disable the collapse. `orfs.gtf` / `orfs_local.gtf` contain all predicted isoforms. |
| `--subseq-terminal-overhang-nt N` | 0   | Nt of overhang allowed at leftmost/rightmost CDS block past the matched keeper block. |
| `--subseq-splice-shift-nt N`      | 0   | Nt of tolerance at each side of an interior CDS block boundary. Must be a multiple of 3 (frame-preserving). |
| `--subseq-allow-exon-skip`        | off | Allow the dropped isoform's exons to be a non-contiguous ordered subset of the keeper's exons. |
| `--subseq-report`                 | off | Also write `orfs.dropped_subseq.tsv` (`dropped_tid`, `keeper_tid`, `reason`). |

## Examples

Short-read short-assembly on a single sample, defaults:

```bash
drusilla annotate \
  --stringtie-gtf stringtie.gtf \
  --genome        GRCh38.fa \
  --out-dir       drusilla_out/
```

Iso-Seq BAM, one CDS per gene, drop noisy transcripts:

```bash
drusilla annotate \
  --bam           isoseq.sorted.bam \
  --genome        GRCh38.fa \
  --longread \
  --min-cov 3 --min-tpm 1 --drop-unstranded --drop-single-exon \
  --single-isoform \
  --out-dir       drusilla_out/
```

Rescue 3'-truncated ORFs into a second GTF:

```bash
drusilla annotate \
  --stringtie-gtf stringtie.gtf \
  --genome        GRCh38.fa \
  --partial-out   drusilla_out/orfs.partial.gtf \
  --out-dir       drusilla_out/
```
