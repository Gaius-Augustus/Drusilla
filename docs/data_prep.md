# Preparing TFRecords for training

Drusilla's `train` command consumes TFRecord shards produced from
labelled transcripts.

## Inputs per species

- A genome FASTA (e.g. `GRCh38.fa`).
- A StringTie GTF of assembled transcripts.
- A reference annotation (GFF3 / GTF).
- A transcript FASTA produced by `gffread -w`.

If you're starting from an RNA-seq BAM, generate the StringTie GTF and
transcript FASTA first:

```bash
stringtie sample.sorted.bam -o stringtie.gtf -p 8
gffread -w transcripts.fa -g GRCh38.fa stringtie.gtf
```

For BAMs that pool runs aligned without a strandedness option, run
`drusilla strand-fix-bam` before StringTie (see
[strand_fix.md](strand_fix.md)).

## Step 1 — label transcripts

For every StringTie transcript, project the reference CDS onto the
transcript and emit a per-position label array
(0=IR, 1=START, 2=E1, 3=E2, 4=E0, 5=STOP).

```bash
python scripts/label_transcripts.py \
  --stringtie-gtf   stringtie.gtf \
  --reference-gff   refseq.gff \
  --transcripts-fa  transcripts.fa \
  --out-dir         labels/<species>/
```

### Label policy v2

This is the policy `drusilla_model` was trained with (Drusilla >= 0.3;
the per-clade models were trained with the earlier policy, see Notes):

- **Unstranded transcripts in both orientations.** A StringTie transcript
  with strand `.` is labelled twice: the forward record (the sequence as
  `gffread` extracts it, matched against `+` reference CDS) and a
  reverse-complement record `<tid>__rc` (matched against `-` reference
  CDS). Both records go into `labels.npz` and `transcripts_labelled.fa`.
  At inference, `--unstranded-both-strands` with `--unstranded-pick
  nonoverlap` is the matching rule. `--single-orientation` restores the
  forward-only labelling.
- **Every gene of a fused transcript.** All complete reference CDS fully
  contained in the transcript are clustered by overlap on the transcript;
  the longest CDS of every cluster is labelled (`kept_fused` when more
  than one cluster, `kept_multi` when one cluster held several
  isoforms). A further gene truncated by the transcript end is labelled
  as far as it is visible (`extra_partial`). At inference,
  `--orfs-per-tx 0` is the matching rule.
- **Incomplete reference CDS stay unlabelled.** RefSeq CDS flagged
  `partial=true` or `pseudo=true` are treated as incomplete even when
  their length is a multiple of 3, so no START is placed on an
  unverified base.

Outputs in `--out-dir`:

- `labels.npz`               — `{record_id: int8[L]}` array set, including
  the `<tid>__rc` records
- `transcripts_labelled.fa`  — the labelled records, i.e. every StringTie
  transcript plus the reverse-complement sequence of every unstranded
  one (the module always emits at least an all-IR array per record)
- `stats.tsv`                — count per label category: `kept_single`,
  `kept_multi`, `kept_fused`, `kept_partial`, `extra_partial`, `ir_only`,
  `antisense_ir`, `ref_partial_ir`, `rc_copies`
- `partial.tsv`              — details for the partially projected
  reference CDS (`kept_partial`, `extra_partial`)

## Step 2 — chunk into TFRecords

Windowed chunks of `chunk_len` (default 9999) with padding on the last
window per transcript:

```bash
python scripts/build_tfrecords.py \
  --fasta   labels/<species>/transcripts_labelled.fa \
  --labels  labels/<species>/labels.npz \
  --out     tfrecords/<species>.tfrecords.gz \
  --chunk-len 9999
```

An output name ending in `.gz` is written GZIP-compressed
(`--compression {auto,none,gzip}` overrides the suffix rule). The one-hot
payload shrinks about 45x; the 425 shards of `drusilla_model` take 21.5 GB
instead of about 1 TB, and the loader reads both kinds transparently.

TFRecord schema per example:

| feature      | dtype   | shape       | notes |
|--------------|---------|-------------|-------|
| `input`      | bytes   | uint8 [L, 6]| A, C, G, T, N, PAD |
| `output`     | bytes   | uint8 [L, 6]| IR, START, E1, E2, E0, STOP one-hot |
| `tx_id`      | bytes   | utf-8       | source record id (`<tid>` or `<tid>__rc`) |
| `chunk_idx`  | int64   | ()          | 0-based chunk index within the record |

Padded positions have `input[..., 5] == 1` and all-zero labels; the
loss layer masks them out.

## Step 3 — write a manifest

The `drusilla train` command reads a TSV where the last column is a
shard path. One line per shard:

```
Homo_sapiens        tfrecords/Homo_sapiens.tfrecords.gz
Mus_musculus        tfrecords/Mus_musculus.tfrecords.gz
Danio_rerio         tfrecords/Danio_rerio.tfrecords
```

Split by species (not by shard) into a training and a validation
manifest; species must not appear in both. With hundreds of shards set
`data.cycle_length: 0` in the config so every shard is read side by side
(see [training.md](training.md)).

## Reproducing the Drusilla model

`drusilla_model` was trained on 425 species of six clades (68 vertebrates,
88 insects, 32 land plants, 213 fungi, 18 diatoms, 6 green algae; the list
is in [model_cfg/drusilla_model.yaml](../model_cfg/drusilla_model.yaml)) with
label policy v2 and gzip shards, validated on 6 held-out species of three
clades, using the full Nextflow pipeline in
[tiberius_orf_finder](https://github.com/Gaius-Augustus/tiberius_orf_finder)
(genome and RNA-seq retrieval, StringTie, labelling, chunking). That
pipeline runs on the group's HPC cluster and is not designed for outside
reproduction; the resulting TFRecord shards are what `drusilla train`
consumed with [configs/drusilla_model.yaml](../configs/drusilla_model.yaml). For a dataset
of your own, follow steps 1-3 above per species and combine the
manifests.

## Notes

- `chunk_len` in the config must equal the `--chunk-len` used when
  writing the TFRecords.
- The label module handles both GTF and GFF3 attribute syntax (`transcript_id
  "..."` and `Parent=...`).
- Antisense-only overlaps are labelled all-IR on purpose — the model
  should learn to reject them.
- Label policy v1 (the per-clade models of Drusilla 0.2, now only `insects`): unstranded
  transcripts were labelled on the forward strand only, only the longest
  contained CDS of a transcript was labelled, and the `partial` / `pseudo`
  flags were ignored. TFRecords built with the current code differ from
  those, so mixing v1 and v2 shards in one training run is not advisable;
  rebuild the v1 shards instead.
- Every StringTie transcript becomes a training example; there is no
  dropping in `label_transcripts.py`.
