# Preparing TFRecords for training

Drusilla's `train` command consumes TFRecord shards produced from
labelled transcripts. There are two paths:

1. **Per-species manual build** (this document). Fine for one species or
   a handful; requires you to bring RNA-seq alignments and a curated
   reference annotation.
2. **Full multi-species data-gen pipeline.** Nextflow-driven; runs
   NCBI-`datasets` genome download, VARUS RNA-seq alignment, StringTie
   assembly, gffcompare labelling, and TFRecord chunking across a species
   list. Lives in the [tiberius_orf_finder](https://github.com/Gaius-Augustus/tiberius_orf_finder)
   sister repo (`nextflow/main_shortread_v2.nf` and friends).

The per-species path is documented here.

## Inputs per species

- A genome FASTA (e.g. `GRCh38.fa`).
- A StringTie GTF of assembled transcripts.
- A reference annotation (GFF3 / GTF) with `CDS` and `stop_codon`
  features (RefSeq / BRAKER / Ensembl all work).
- A transcript FASTA produced by `gffread -w`.

If you're starting from an RNA-seq BAM, generate the StringTie GTF and
transcript FASTA first:

```bash
stringtie sample.sorted.bam -o stringtie.gtf -p 8
gffread -w transcripts.fa -g GRCh38.fa stringtie.gtf
```

## Step 1 — label transcripts

For every StringTie transcript, project the reference CDS onto the
transcript and emit a per-position int8 label array
(0=IR, 1=START, 2=E1, 3=E2, 4=E0, 5=STOP).

```bash
python scripts/label_transcripts.py \
  --stringtie-gtf   stringtie.gtf \
  --reference-gff   refseq.gff \
  --transcripts-fa  transcripts.fa \
  --out-dir         labels/<species>/
```

Outputs in `--out-dir`:

- `labels.npz`               — `{transcript_id: int8[L]}` array set
- `transcripts_labelled.fa`  — subset of the input FASTA containing only
  the transcripts that were labelled (i.e. every StringTie transcript;
  the module always emits at least an all-IR array).
- `stats.tsv`                — count of each label category
  (`kept_single`, `kept_multi`, `kept_partial`, `ir_only`,
  `antisense_ir`, `ref_partial_ir`).
- `partial.tsv`              — details for the `kept_partial` cases.

## Step 2 — chunk into TFRecords

Windowed chunks of `chunk_len` (default 9999) with padding on the last
window per transcript:

```bash
python scripts/build_tfrecords.py \
  --fasta   labels/<species>/transcripts_labelled.fa \
  --labels  labels/<species>/labels.npz \
  --out     tfrecords/<species>.tfrecords \
  --chunk-len 9999
```

TFRecord schema per example:

| feature      | dtype   | shape       | notes |
|--------------|---------|-------------|-------|
| `input`      | bytes   | uint8 [L, 6]| A, C, G, T, N, PAD |
| `output`     | bytes   | uint8 [L, 6]| IR, START, E1, E2, E0, STOP one-hot |
| `tx_id`      | bytes   | utf-8       | source transcript id |
| `chunk_idx`  | int64   | ()          | 0-based chunk index within the transcript |

Padded positions have `input[..., 5] == 1` and all-zero labels; the
loss layer masks them out.

## Step 3 — write a manifest

The `drusilla train` command reads a TSV where the last column is a
shard path. One line per shard:

```
Homo_sapiens        tfrecords/Homo_sapiens.tfrecords
Mus_musculus        tfrecords/Mus_musculus.tfrecords
Danio_rerio         tfrecords/Danio_rerio.tfrecords
```

Split by species (not by shard) into a training and a validation
manifest; species must not appear in both.

## Reproducing the released vertebrate model

The vertebrate release was trained on 51 species with 4 held-out
validation species using the full Nextflow pipeline in
[tiberius_orf_finder](https://github.com/Gaius-Augustus/tiberius_orf_finder).
That pipeline runs on the group's HPC cluster and is not designed for
outside reproduction; the resulting TFRecord shards are what would be
handed to `drusilla train`. If you want a comparable dataset for your
own clade, follow steps 1-3 above per species and combine the manifests.

## Notes

- `chunk_len` in the config must equal the `--chunk-len` used when
  writing the TFRecords.
- The label module handles both GTF and GFF3 attribute syntax (`transcript_id
  "..."` and `Parent=...`).
- Antisense-only overlaps are labelled all-IR on purpose — the model
  should learn to reject them.
- Every StringTie transcript becomes a training example; there is no
  dropping in `label_transcripts.py`.
