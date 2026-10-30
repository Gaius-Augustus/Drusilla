# `drusilla strand-fix-bam` — reference

Recover library strandedness in a pooled RNA-seq BAM before running
StringTie.

Pooled SRA runs aligned without a strandedness option (e.g. VARUS /
HISAT2 defaults) carry an `XS` strand tag only on spliced reads. StringTie
then reports strand `.` for most single-exon transcripts and cannot
separate overlapping antisense genes, even if the underlying runs were
stranded. This command infers the protocol of each run and sets `XS:A` on
all mapped reads of stranded runs.

Requires `pysam`: `pip install "drusilla[bam]"` (also part of `[gpu]` and
of the container image).

## How it works

1. Pass 1: for every run, count spliced reads that carry `XS` and check
   whether the first read of the fragment maps to the `XS` strand.
   Secondary, supplementary, duplicate and QC-fail reads are not counted.
2. Each run is called:
   - `FR`: read1 maps to the transcript strand (fraction of agreement
     at or above `--stranded-threshold`).
   - `RF`: read1 maps antisense to the transcript (fraction at or below
     `1 - --stranded-threshold`).
   - `unstranded`: fraction in between.
   - `unknown`: fewer than `--min-spliced` spliced reads.
3. Pass 2: the BAM is rewritten. Every mapped read of an `FR` / `RF` run
   gets `XS:A` set from its fragment orientation, unspliced reads
   included. Reads of unstranded and unknown runs are written unchanged
   (or dropped with `--drop-unstranded-runs`).

The run id is taken from the read name with `--run-regex`; group 1 is the
id. The default matches leading letters followed by digits, so
`SRR123456.789` gives `SRR123456`. Reads that do not match are grouped as
`_unmatched`.

## Usage

```bash
drusilla strand-fix-bam \
  --in-bam      aligned.bam \
  --out-bam     aligned.strandfix.bam \
  --report-tsv  strand_runs.tsv

stringtie aligned.strandfix.bam -o stringtie.gtf
```

The same step runs inside `drusilla annotate` with `--bam ... --fix-strand`
(StringTie then runs on the fixed BAM; the fixed BAM is deleted unless
`--keep-tmp`, `strand_runs.tsv` is kept in `--out-dir`).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--in-bam FILE`              | required | Input BAM (any order; read once per pass). |
| `--out-bam FILE`             | required | Output BAM with `XS:A` set on all mapped reads of stranded runs. |
| `--report-tsv FILE`          | required | Per-run report (see below). |
| `--run-regex RE`             | leading letters+digits | Regex whose group 1 is the run id in the read name. `none`: the whole BAM is one run (a BAM of a single library). |
| `--min-spliced N`            | 2000 | Spliced reads with `XS` needed to call a run; fewer gives `unknown`. |
| `--stranded-threshold F`     | 0.9  | Agreement fraction for a stranded call; runs between `1-F` and `F` are `unstranded`. |
| `--max-pass1-reads N`        | all  | Stop pass 1 after N records (speed). |
| `--drop-unstranded-runs`     | off  | Do not write reads of unstranded / unknown runs. |
| `--threads N`                | 4    | BAM (de)compression threads. |
| `--no-index`                 | off  | Skip indexing the output BAM. |

## Report TSV

One row per run, sorted by read count (descending), plus a final
`#total` line.

| Column | Meaning |
|---|---|
| `run`              | Run id. |
| `n_spliced`        | Spliced reads with `XS` seen in pass 1. |
| `frac_read1_eq_xs` | Fraction of those whose read1 orientation equals `XS` (`NA` if none). |
| `call`             | `FR`, `RF`, `unstranded` or `unknown`. |
| `n_reads`          | Reads of the run seen in pass 2. |
| `n_tagged`         | Reads that received `XS:A`. |
| `n_dropped`        | Reads dropped (`--drop-unstranded-runs`). |

The `#total` line holds only the `n_reads`, `n_tagged` and `n_dropped`
sums.
