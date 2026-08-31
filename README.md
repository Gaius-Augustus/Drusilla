# Drusilla

Deep-learning ORF annotator for RNA-seq assembled transcripts. Given a
StringTie GTF (or a sorted RNA-seq BAM) plus the corresponding genome,
Drusilla predicts the coding region of every assembled transcript and
writes a genomic GTF of CDS lines.

Under the hood: CNN + BiLSTM per-position classifier over
`IR / START / E1 / E2 / E0 / STOP`, followed by a 6-state structured HMM
that enforces ATG starts, in-frame stops, and a single canonical reading
frame per ORF.

> **Model availability.** Only the **vertebrate** model is released at
> the moment (`--model vertebrates`, the default). Models for
> other clades are in training and will be added to
> [`models.yaml`](models.yaml) as they become available. See the
> [Models](#models) section for the current release table.

---

## Install

```bash
pip install "drusilla[gpu] @ git+https://github.com/Gaius-Augustus/Drusilla"
```

The `[gpu]` extra pulls in TensorFlow, `bricks2marble`, `hidten`,
`biopython`, and `pandas`. Without it you get only the CLI skeleton and
the pure-Python data helpers; the annotate / train commands will fail to
import TensorFlow.

External tools required on `PATH`:

- `gffread` — always (extracts transcript sequences from the genome)
- `stringtie` — only if you pass a BAM (StringTie is run internally)

A Singularity image bundling all three will be released alongside this
package.

---

## Quick start

Annotate a transcriptome from an existing StringTie GTF (weights are
downloaded and cached on first use):

```bash
drusilla annotate \
  --stringtie-gtf path/to/stringtie.gtf \
  --genome        path/to/genome.fa \
  --model         vertebrates \
  --out-dir       results/
```

From a sorted RNA-seq BAM (StringTie is run internally):

```bash
drusilla annotate \
  --bam           path/to/aligned.sorted.bam \
  --genome        path/to/genome.fa \
  --out-dir       results/
```

For PacBio Iso-Seq / ONT spliced BAMs add `--longread` (passes `-L` to
StringTie).

Output (in `--out-dir`) by default:

- `orfs.gtf` — **primary**, genomic-coordinate CDS lines
  (1-based inclusive, `source = "drusilla"`, one CDS line per genomic
  sub-interval of every predicted ORF, shared `transcript_id` /
  `gene_id` per ORF), subsequence-isoform collapse applied.
- `orfs_local.gtf` — same predictions in transcript coordinates.
- `orfs.log` — b2m annotation log.

Intermediate StringTie / gffread output is deleted unless `--keep-tmp`
is set.

See [docs/annotate.md](docs/annotate.md) for the full flag reference.


## Models

Only the **vertebrate** model is released today. Other clades are in
training; this table will grow as they land in [`models.yaml`](models.yaml).

| Name (`--model`)   | Clade / target      | Status         | Architecture       | Notes |
|--------------------|---------------------|----------------|--------------------|-------|
| `vertebrates`      | Vertebrata          | **Released**   | CNN + BiLSTM (6-state HMM head) | Default. Trained on 51 vertebrate species; validated on 4 held-out species. |
| `embryophyta`      | Land plants         | In training    | —                  | Planned. |
| `fungi`            | Fungi               | In training    | —                  | Planned. |
| `insecta`          | Insects             | Planned        | —                  | — |

Applying a released model to species from another clade is technically
possible but the predictions are not benchmarked and will underperform.
Wait for the matching clade release, or train your own with
`drusilla train` (see [docs/training.md](docs/training.md)).

Model releases are described by per-model manifests under
[`model_cfg/`](model_cfg/) (one YAML per model — see
[`model_cfg/README.md`](model_cfg/README.md) for the schema and how to
publish a new release). Each manifest points at a `.tar.gz` archive
that bundles both the weights and their architecture config, so
released models are always internally consistent.

List available models and their local cache status:

```bash
drusilla models list
```

Download a specific model without running annotate:

```bash
drusilla models download vertebrates
```

Downloaded archives are verified against the manifest's SHA256, then
extracted to `$XDG_CACHE_HOME/drusilla/models/<name>-v<version>/`
(default: `~/.cache/drusilla/models/`). Set `DRUSILLA_CACHE_DIR` to
override the cache location, or `DRUSILLA_MODEL_CFG_DIR` to add a
directory of extra manifests (useful for staging a new release without
editing the package).

---

## Training

Given a prepared TFRecord dataset, train a model:

```bash
drusilla train \
  --train-manifest results/tfrecords/train_manifest.tsv \
  --val-manifest   results/tfrecords/val_manifest.tsv \
  --config         configs/vertebrates.yaml \
  --out-dir        results/models/run_001
```

Manifests are two-column TSVs: `species<TAB>path/to/shard.tfrecords`
(one line per shard). See [docs/training.md](docs/training.md) for full
details.

Preparing the TFRecords from raw genomes + reference annotations is a
larger data-generation pipeline, described separately in
[docs/data_prep.md](docs/data_prep.md).

---


## License

MIT — see [LICENSE](LICENSE).
