# Drusilla

Deep-learning ORF annotator for RNA-seq assembled transcripts. Given a
StringTie GTF (or a sorted RNA-seq BAM) plus the corresponding genome,
Drusilla predicts the coding region of every assembled transcript and
writes a genomic GTF of CDS lines.

Under the hood: CNN + BiLSTM per-position classifier over
`IR / START / E1 / E2 / E0 / STOP`, followed by a 6-state structured HMM
that enforces ATG starts, in-frame stops, and a single canonical reading
frame per ORF.

Currently only a **vertebrate** model is released. Additional clade
models will follow.

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

Output (in `--out-dir`):

- `orfs.gtf` — the predicted CDS annotation (1-based inclusive
  coordinates, `source = "drusilla"`, one CDS line per genomic
  sub-interval of every predicted ORF, shared `transcript_id` /
  `gene_id` per ORF).
- `orfs.log` — b2m annotation log.
- Intermediate StringTie / gffread output (deleted unless
  `--keep-tmp` is set).

See [docs/annotate.md](docs/annotate.md) for the full flag reference.

---

## Models

Model weights are hosted separately and downloaded on demand. Available
models and their local cache status:

```bash
drusilla models list
```

Download a specific model without running annotate:

```bash
drusilla models download vertebrates
```

Weights are cached under `$XDG_CACHE_HOME/drusilla/models/`
(default: `~/.cache/drusilla/models/`). Set `DRUSILLA_CACHE_DIR` to
override.

To point Drusilla at a different registry (for testing new releases or
using a mirror), set `DRUSILLA_MODELS_URL` to a URL that returns a YAML
document with the same schema as the bundled `models.yaml`.

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

## Citation

If you use Drusilla, please cite:

> Gabriel L. et al. Drusilla: deep-learning ORF annotator for
> RNA-seq assembled transcripts. In preparation.

## License

MIT — see [LICENSE](LICENSE).
