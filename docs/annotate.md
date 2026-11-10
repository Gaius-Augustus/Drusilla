# `drusilla annotate` — reference

End-to-end ORF annotation from a transcript assembly GTF (StringTie or any
other assembler) or a sorted RNA-seq BAM.

## Input modes

Exactly one is required:

| Flag | Meaning |
|---|---|
| `--assembly-gtf FILE` | GTF of assembled transcripts: `exon` lines carrying `transcript_id` and `gene_id`, from StringTie or any other assembler. StringTie is not run. The former spelling `--stringtie-gtf` is still accepted (hidden; prints a deprecation note). |
| `--bam FILE`           | Run StringTie internally on this sorted BAM. |

Both modes also require `--genome FILE` (the genome FASTA whose contig
names match the GTF/BAM).

If you already have a spliced transcript FASTA (e.g. from a previous
`gffread` run), you can pass `--transcripts-fa FILE` to skip that step
too. The FASTA record IDs must equal the GTF `transcript_id`s.

## Model selection

Exactly one path chooses the model:

| Flag | Meaning |
|---|---|
| _(default)_             | Uses the default registered model (`DEFAULT_MODEL` in `drusilla/cli/annotate.py`): `drusilla_model`, whose weights ship with the package. |
| `--model NAME`          | Registered model (see `drusilla models list`). `drusilla_model` is bundled: weights and architecture config are read from the package and the SHA256 of the weights is checked. No `--config` needed. |
| `--weights FILE`        | Local `.weights.h5`. Requires `--config FILE` alongside it (the architecture is not carried inside `.weights.h5`). |

If you pass `--config` together with `--model`, the local config
overrides the arch config the registry provides for the model. Only useful
when you know exactly what you're doing.

**Inference presets from the manifest.** A registered model's manifest
may carry an `annotate_defaults` block for the options
`unstranded_both_strands`, `unstranded_pick`, `orfs_per_tx`, `flank_bp`,
`flank_3p_bp` and `flank_keep`. They fill in whatever you did not set on
the command line; explicit flags always win (the boolean options have
`--no-...` forms for that). The applied presets are printed as
`[model] inference defaults taken from the model manifest: ...`. The
`drusilla_model` manifest presets both-strand scanning with `nonoverlap`,
`orfs_per_tx 0` and 500 bp flanks with completion, matching its training
labels (label policy v2). With
`--weights` no manifest is involved and the built-in defaults apply.

> **Available model: `drusilla_model` (default, bundled) for vertebrates,
> insects, land plants, fungi, diatoms and green algae.** See the
> [Models table in the README](../README.md#models).

## Output

Written under `--out-dir` in the default mode:

| File | Description |
|---|---|
| `orfs.gtf`       | **PRIMARY.** Genomic-coordinate GTF, one `CDS` line per genomic sub-interval of every predicted ORF (1-based inclusive; `source = "drusilla"`; shared `transcript_id` / `gene_id` per ORF). Subsequence-isoform collapse is applied by default (see below). |
| `orfs_local.gtf` | Same predictions in transcript (local) coordinates: one `CDS` line per predicted ORF, contig column holds the sequence record (`<tid>`, or `<tid>__rc` for the reverse-complement copy of an unstranded transcript), strand = `+`, phase of the first written base (non-zero only for ORFs completed into the 5' flank, whose flank part is clipped). The part of a completed ORF that lies in a genomic flank is clipped here; the `completion` attribute marks such ORFs. Always in sync with `orfs.gtf` (same set of kept ORFs). |
| `orfs.log`       | b2m annotation log. |
| `stringtie.gtf`  | Written when starting from `--bam`. Kept. |
| `strand_runs.tsv` | Per-run strandedness report, written with `--fix-strand` (see [strand_fix.md](strand_fix.md)). Kept. |

ORFs that were completed into a genomic flank (`--flank-keep`) carry the
GTF attribute `completion "5p"`, `"3p"` or `"5p3p"`. Additional ORFs on
the same transcript (`--orfs-per-tx`) get ids `<tid>.orf2`, `<tid>.orf3`,
... and their own `gene_id`.

Extra outputs, only produced when their flag is set:

| File | Enabled by |
|---|---|
| `orfs.dropped_subseq.tsv` | `--subseq-report` |
| `<partial-out.gtf>`       | `--partial-out FILE`  (3'-truncated ORFs) |
| `<partial5-out.gtf>`      | `--partial5-out FILE` (5'-truncated ORFs) |
| `stringtie.filtered.gtf`  | Any of `--min-cov / --min-tpm / --drop-unstranded / --drop-single-exon`. |
| `transcripts.fa`, `transcripts.bothstrands.fa`, padded FASTAs (`transcripts.pad*.fa`, `transcripts.flank*.fa`), `orfs.b2m_local.gtf` (b2m's own GTF), `aligned.strandfix.bam` | Intermediates. Deleted unless `--keep-tmp` is set. |

## Flags

### Inference

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size N`         | auto | NN inference batch size. If omitted, auto-selected from the GPU memory (queried via `nvidia-smi`) and the model's `chunk_len`. Empirical anchors: 40 GB A100 -> ~180, 80 GB A100 -> ~360, smaller GPUs scale down. If `nvidia-smi` is unavailable the fallback is 8. Override any time with an explicit value. |
| `--parallel N`           | 100 | HMM parallel-scan factor (chunks padded to a multiple of N). |
| `--no-codon-emitter`     | off | Drop the hard ATG-at-START / not-stop-at-E2 / stop-at-STOP codon constraints (debug). |
| `--restrict-start-to-ir-start` | off | Force the HMM's initial state distribution to `{IR, START}` (whole-transcript decode should never start mid-codon). |
| `--ir-prior FLOAT`       | 0.5 | Under `--restrict-start-to-ir-start`, `P(start in IR)`. |
| `--threads N`            | 4   | Threads for StringTie and for BAM (de)compression under `--fix-strand` (gffread is single-threaded). |
| `--longread`             | off | Pass `-L` to StringTie (PacBio Iso-Seq / ONT). |
| `--keep-tmp`             | off | Do not delete intermediate FASTA / BAM files and b2m's GTF. |
| `--fix-strand`           | off | With `--bam`: recover per-run library strandedness first and run StringTie on the fixed BAM. Needs `pysam`. Writes `strand_runs.tsv`. See [strand_fix.md](strand_fix.md). |
| `--strand-run-regex RE`  | leading letters+digits | With `--fix-strand`: regex whose group 1 is the run id in a read name (e.g. `SRR123456`). `none`: the whole BAM is one library. |

### Genomic flanks and ORF completion

Adds context around each transcript so the model has a left window at
position 0 and, optionally, so ORFs cut off by an incomplete assembly
can be completed from the genome.

| Flag | Default | Meaning |
|---|---|---|
| `--prefix-pad-n K`       | 0 | Prepend K literal Ns per transcript. ORFs starting inside the pad are dropped. Excludes `--flank-bp` and `--flank-3p-bp`. |
| `--flank-bp K`           | 0 (manifest may preset) | Prepend K bp of real upstream genomic context (RC for `-` strand); short chromosome edges are N-padded. Mutually exclusive with `--prefix-pad-n`. |
| `--flank-3p-bp K`        | 0 (manifest may preset) | Append K bp of real downstream genomic context. Only useful with `--flank-keep`: an ORF whose STOP falls in the 3' flank is then emitted as a completed CDS. |
| `--flank-clip`           | off | With `--flank-bp`, clip ORFs whose START falls in the flank to the next in-frame codon inside the transcript. Without this (and without `--flank-keep`), such ORFs are dropped. Mutually exclusive with `--flank-keep`. |
| `--flank-keep` / `--no-flank-keep` | off (manifest may preset) | With `--flank-bp` and/or `--flank-3p-bp`, keep ORFs that extend into a flank and project the flank part to the genome (ORF completion). Completed ORFs carry the attribute `completion "5p\|3p\|5p3p"`. |
| `--flank-min-tx-frac F`  | 0.5 | With `--flank-keep`: minimum fraction of an ORF's coding bases that must lie inside the assembled transcript; ORFs below it are dropped. Must be between 0 and 1. |

### Unstranded transcripts and several ORFs per transcript

Transcripts with strand `.` are modelled on the forward strand,
because gffread extracts them as the forward genomic sequence.

| Flag | Default | Meaning |
|---|---|---|
| `--unstranded-both-strands` / `--no-unstranded-both-strands` | off (manifest may preset; `drusilla_model`: on) | Also scan the reverse complement of every strand-`.` transcript as a separate record `<tid>__rc` with a `-` strand model. Without it, minus-strand genes on unstranded (mostly single-exon) transcripts cannot be called. |
| `--unstranded-pick {nonoverlap,longest,both}` | `longest` (manifest may preset; `drusilla_model`: `nonoverlap`) | With `--unstranded-both-strands`: `nonoverlap` pools the ORFs of both orientations, keeps them longest first and drops every ORF that shares genomic bases with a kept one (a gene called in both orientations keeps the longer call, genes on different strands of one transcript survive); `longest` keeps only the orientation with the longest ORF; `both` keeps everything. `nonoverlap` is the rule matching label policy v2, under which `drusilla_model` was trained. |
| `--orfs-per-tx N`           | 1 (manifest may preset; `drusilla_model`: 0) | Maximum ORFs emitted per transcript, longest first. Extra ORFs get ids `<tid>.orf2`, `<tid>.orf3`, ... and their own `gene_id`. 0 = no limit. Needed for fused transcripts carrying several genes; the training labels of `drusilla_model` contain every gene of a fused transcript. |

### Reference-free assembly pre-filters

Approximate the "clean-transcript" curation the training pipeline does
with a reference annotation. `--min-cov` and `--min-tpm` read the
StringTie attributes `cov` and `TPM`; a transcript without the attribute
counts as 0 and is dropped, so use them on a GTF from another assembler
only if it writes these attributes.

| Flag | Meaning |
|---|---|
| `--min-cov FLOAT`        | Drop transcripts whose `cov` attribute is below this. |
| `--min-tpm FLOAT`        | Drop transcripts whose `TPM` attribute is below this. |
| `--drop-unstranded`      | Drop transcripts with strand `.`. |
| `--drop-single-exon`     | Drop single-exon transcripts (often intronic noise). |

Typical short-read defaults: `--min-cov 3 --min-tpm 1 --drop-unstranded`.

### Post-decode ORF filters

| Flag | Default | Meaning |
|---|---|---|
| `--min-coding-length N`  | 200 | Drop ORFs whose total CDS length is below N nt. Set to 0 to disable. |
| `--single-isoform`       | off | Keep at most one transcript per StringTie `gene_id` — the one with the longest predicted CDS (ties broken by `transcript_id`). |
| `--min-utr-5 N`          | 0   | Drop ORFs whose START is within N nt of the transcript 5' end (proxy for 5' assembly truncation). Not applied to an end that was completed into a flank. |
| `--min-utr-3 N`          | 0   | Drop ORFs whose STOP is within N nt of the transcript 3' end. Not applied to an end that was completed into a flank. |

### Optional partial-ORF output

| Flag | Meaning |
|---|---|
| `--partial-out FILE`     | Also write 3'-truncated ORFs (`START ... E*`, no STOP within the transcript). |
| `--partial5-out FILE`    | Also write 5'-truncated ORFs (`E*` at position 0, ending at a STOP; no upstream ATG). |
| `--lorf-class`           | Add a `lorf_class` GTF attribute per ORF (`LORF_UPSTOP` / `sORF_UPSTOP` / `upLORF` / `LORF_NOUPSTOP`). Requires loading transcript sequences into memory. |

### Subsequence-isoform collapse (default ON)

Drops predicted isoforms whose spliced CDS is a (near-)sub-sequence of
another isoform in the same StringTie locus (locus = `transcript_id`
minus the `.orfN` / `__rc` suffixes and the trailing `.<iso>`). Applied automatically to both `orfs.gtf`
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
  --assembly-gtf  stringtie.gtf \
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

Pooled SRA BAM, recover strandedness, scan both strands of unstranded
transcripts and complete truncated ORFs from the genome (with
the default `drusilla_model` the last three lines are its manifest presets and
can be dropped):

```bash
drusilla annotate \
  --bam           aligned.sorted.bam \
  --genome        genome.fa \
  --fix-strand \
  --unstranded-both-strands --unstranded-pick nonoverlap \
  --flank-bp 500 --flank-3p-bp 500 --flank-keep \
  --orfs-per-tx 0 \
  --out-dir       drusilla_out/
```

`drusilla_model`, but one ORF per transcript and no flanks (overriding its
manifest presets):

```bash
drusilla annotate \
  --assembly-gtf  stringtie.gtf \
  --genome        genome.fa \
  --orfs-per-tx 1 --flank-bp 0 --flank-3p-bp 0 --no-flank-keep \
  --out-dir       drusilla_out/
```

Rescue 3'-truncated ORFs into a second GTF:

```bash
drusilla annotate \
  --assembly-gtf  stringtie.gtf \
  --genome        GRCh38.fa \
  --partial-out   drusilla_out/orfs.partial.gtf \
  --out-dir       drusilla_out/
```
