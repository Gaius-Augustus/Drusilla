# Drusilla

Deep-learning ORF annotator for RNA-seq assembled transcripts. Given a
GTF of assembled transcripts (StringTie or any other assembler; or a
sorted RNA-seq BAM, on which StringTie is run) plus the corresponding genome,
Drusilla predicts the coding region of every assembled transcript and
writes a genomic GTF of CDS lines.

[Paludamentum](https://github.com/Gaius-Augustus/Paludamentum) runs Drusilla in a Nextflow pipeline that combines it with ab initio predictions from [Tiberius](https://github.com/Gaius-Augustus/Tiberius).

> **Model availability.** Drusilla ships one model, `drusilla_model`, for
> vertebrates, insects, land plants, fungi, diatoms and green algae. Its
> weights are part of the repository and of the Python package, so nothing
> is downloaded and `--model` can be omitted. The per-clade models of
> Drusilla 0.2 (`--model vertebrates`, `--model insects`) stay registered
> as legacy downloads. See the [Models](#models) section.

---

## Install

```bash
pip install "drusilla[gpu] @ git+https://github.com/Gaius-Augustus/Drusilla"
```

The `[gpu]` extra pulls in TensorFlow, `bricks2marble`, `hidten`,
`biopython`, `pandas`, and `pysam`. Without it you get only the CLI skeleton and
the pure-Python data helpers; the annotate / train commands will fail to
import TensorFlow.

External tools required on `PATH`:

- `gffread` — always (extracts transcript sequences from the genome)
- `stringtie` — only if you pass a BAM (StringTie is run internally)

Optional Python dependency: `pysam`, needed only for `--fix-strand` and
`drusilla strand-fix-bam` (`pip install "drusilla[bam]"`; already part of
`[gpu]` and of the container).

A Singularity image bundling all three is published on Docker Hub as
[`larsgabriel23/drusilla`](https://hub.docker.com/r/larsgabriel23/drusilla).
See [Container](#container) below.

---

## Container

A prebuilt image is available at
`docker://larsgabriel23/drusilla:latest`. It is based on the NVIDIA NGC
TensorFlow image (`nvcr.io/nvidia/tensorflow:25.02-tf2-py3`) and bundles
Drusilla, `gffread`, and `stringtie`.

### Singularity / Apptainer

Build a local `.sif` from the Docker Hub image:

```bash
singularity build drusilla.sif docker://larsgabriel23/drusilla:latest
# or, with Apptainer:
apptainer build drusilla.sif docker://larsgabriel23/drusilla:latest
```

Run it (bind-mount your working dir and a cache dir for model weights):

```bash
singularity run --nv \
  -B $PWD:/work \
  -B $HOME/.cache/drusilla:/data/drusilla_cache \
  drusilla.sif annotate \
    --assembly-gtf  /work/stringtie.gtf \
    --genome        /work/genome.fa \
    --out-dir       /work/results
```

`--nv` is required for GPU access. The bundled `drusilla_model` needs no
download; the second bind mount only matters for the legacy per-clade
models, whose archives are downloaded into `DRUSILLA_CACHE_DIR`
(`/data/drusilla_cache` inside the image).

On clusters that block direct DockerHub pulls from compute nodes, build
the `.sif` once on a login node (or a workstation) and copy it to the
cluster.

### Docker

```bash
docker run --rm --gpus all \
  -v $PWD:/work \
  -v $HOME/.cache/drusilla:/data/drusilla_cache \
  larsgabriel23/drusilla:latest annotate \
    --assembly-gtf  /work/stringtie.gtf \
    --genome        /work/genome.fa \
    --out-dir       /work/results
```

### Building the image

[docker/build_push.sh](docker/build_push.sh) builds
[docker/Dockerfile](docker/Dockerfile) from the checked-out source tree,
so the image contains exactly the commit it was built from (recorded in
the OCI label `org.opencontainers.image.revision`), tags it
`v<version>` (from `pyproject.toml`) and `latest`, and pushes both:

```bash
docker/build_push.sh                 # build + push larsgabriel23/drusilla
PUSH=0 docker/build_push.sh          # build only
IMAGE=gaiusaugustus/drusilla docker/build_push.sh
```

It refuses a dirty working tree (`ALLOW_DIRTY=1` overrides) and takes
`DOCKER` (e.g. `sudo docker`) and `TAG` from the environment.

---

## Quick start

Annotate a transcriptome from an existing assembly GTF, from StringTie or
any other assembler (the bundled `drusilla_model` is used, nothing is
downloaded):

```bash
drusilla annotate \
  --assembly-gtf  path/to/stringtie.gtf \
  --genome        path/to/genome.fa \
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

If the BAM pools several sequencing runs that were aligned without a
strandedness option (e.g. VARUS / HISAT2 defaults), add `--fix-strand`
(requires `--bam`). It infers each run's protocol and sets the `XS` tag
on all reads of stranded runs before StringTie runs, so single-exon
transcripts get a strand. Details in
[docs/strand_fix.md](docs/strand_fix.md).

Output (in `--out-dir`) by default:

- `orfs.gtf` — **primary**, genomic-coordinate CDS lines
  (1-based inclusive, `source = "drusilla"`, one CDS line per genomic
  sub-interval of every predicted ORF, shared `transcript_id` /
  `gene_id` per ORF), subsequence-isoform collapse applied.
- `orfs_local.gtf` — same predictions in transcript coordinates.
- `orfs.log` — b2m annotation log.
- `stringtie.gtf` — kept when starting from a BAM.
- `strand_runs.tsv` — per-run strandedness report, with `--fix-strand`.

Intermediate FASTA / BAM files (and b2m's own GTF) are deleted unless
`--keep-tmp` is set.

See [docs/annotate.md](docs/annotate.md) for the full flag reference.


## Unstranded, truncated and fused transcripts

StringTie leaves single-exon transcripts of unstranded libraries without a
strand (`.`). Drusilla models these on the forward strand, as `gffread`
extracts them. Three optional mechanisms recover genes that such
assemblies otherwise miss (see [docs/annotate.md](docs/annotate.md) for
details and [docs/strand_fix.md](docs/strand_fix.md) for the BAM-level
alternative):

| Flag | Effect |
|---|---|
| `--unstranded-both-strands` | Also scan the reverse complement of every `.` transcript (record `<tid>__rc`). `--unstranded-pick` decides what is kept: `nonoverlap` keeps the ORFs of both orientations, longest first, minus those overlapping a kept ORF on the genome (the rule matching the training labels of `drusilla_model`); `longest` keeps only the orientation with the longest ORF; `both` keeps everything. Needed for minus-strand genes on unstranded single-exon transcripts. |
| `--flank-bp K --flank-3p-bp K --flank-keep` | Add `K` bp of genomic context on both sides of each transcript and keep ORFs whose start or stop falls in that flank ("ORF completion" of 5'/3'-truncated assemblies). Completed ORFs carry the GTF attribute `completion` (`5p`, `3p` or `5p3p`); `--flank-min-tx-frac` (default 0.5) limits how much of an ORF may lie outside the transcript. |
| `--orfs-per-tx N` | Emit up to `N` ORFs per transcript, longest first (`0` = all). Extra ORFs get ids `<tid>.orf2`, `<tid>.orf3`, ... Needed for fused transcripts that carry several genes. |
| `--fix-strand` (with `--bam`) | Recover per-run library strandedness in a pooled BAM before StringTie, so fewer `.` transcripts are assembled in the first place. |

The built-in defaults are the conservative ones (forward strand only,
`longest`, one ORF per transcript, no flanks). A model's manifest can
preset these options (`annotate_defaults`, see
[`model_cfg/README.md`](model_cfg/README.md)): `drusilla_model` turns on
both-strand scanning with `nonoverlap`, all ORFs per transcript and
500 bp flanks with ORF completion, because that is how its training
labels were built and how it was benchmarked. Anything given on the
command line wins (`--no-unstranded-both-strands`, `--orfs-per-tx 1`,
`--flank-bp 0` ...). The legacy `vertebrates` and `insects` manifests set
nothing, so their behaviour is unchanged.

Example for a compact genome assembled from pooled, partly unstranded
RNA-seq; with `drusilla_model` the four option lines are its manifest
presets and can be dropped, with a legacy per-clade model they are needed:

```bash
drusilla annotate \
  --bam           pooled.sorted.bam --fix-strand \
  --genome        genome.fa \
  --unstranded-both-strands --unstranded-pick nonoverlap \
  --flank-bp 500 --flank-3p-bp 500 --flank-keep \
  --orfs-per-tx 0 \
  --out-dir       results/
```

## Models

Drusilla ships one model, `drusilla_model`, trained on 425 species of six
clades. Its weights (`models/drusilla_model/model.weights.h5`, 28 MB) and
architecture config (`configs/drusilla_model.yaml`) are part of the
repository and of the wheel; the manifest
[`model_cfg/drusilla_model.yaml`](model_cfg/drusilla_model.yaml) points at
them with `weights_path` / `arch_path`, and the SHA256 of the weights file
is checked on use. The per-clade models of Drusilla 0.2 stay registered as
legacy downloads.

| Name (`--model`)   | Clade / target      | Status         | Architecture       | Notes |
|--------------------|---------------------|----------------|--------------------|-------|
| `drusilla_model`   | Vertebrata, Insecta, Embryophyta, Fungi, diatoms, Chlorophyta | **Released, default, bundled** | CNN + BiLSTM (6-state HMM head) | Trained on 425 species with label policy v2 ([docs/data_prep.md](docs/data_prep.md)), validated on 6 held-out species of 3 clades; epoch 73 of training run `cnn_lstm_joint_run001`. In the test benchmarks it matched or beat every per-clade model (mean gene-level F1 +2.9, land plants +8.8). |
| `vertebrates`      | Vertebrata          | Legacy (download) | CNN + BiLSTM (6-state HMM head) | Default of Drusilla 0.2. Trained on 51 vertebrate species; validated on 4 held-out species. |
| `insects`          | Insecta             | Legacy, preliminary (download) | CNN + BiLSTM (6-state HMM head) | Trained on the 99 insect species listed in the manifest. |

What distinguishes `drusilla_model` from the legacy models, apart from the
clades it covers: its manifest presets both-strand scanning of unstranded
transcripts with the `nonoverlap` rule, no cap on ORFs per transcript and
500 bp genomic flanks with ORF completion (see
[Unstranded, truncated and fused transcripts](#unstranded-truncated-and-fused-transcripts)),
so `drusilla annotate` needs none of those flags. Its training data follow
label policy v2, so its predictions can contain several genes per fused
transcript and minus-strand genes on unstranded transcripts, which the
per-clade models never emit.

Applying a legacy per-clade model to species from another clade is
technically possible but the predictions are not benchmarked and will
underperform; use `drusilla_model`, or train your own with
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

Download a legacy per-clade model without running annotate
(`drusilla_model` is bundled and needs no download):

```bash
drusilla models download vertebrates
```

Downloaded archives are verified against the manifest's SHA256, then
extracted to `$XDG_CACHE_HOME/drusilla/models/<name>/` (default:
`~/.cache/drusilla/models/`). A `.drusilla_version` marker in that
directory records the manifest version, so bumping `version` in a
manifest triggers a re-download. Set `DRUSILLA_CACHE_DIR` to
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
  --config         configs/drusilla_model.yaml \
  --out-dir        results/models/run_001
# after a walltime limit: same command plus --resume
```

Manifests are two-column TSVs: `species<TAB>path/to/shard.tfrecords[.gz]`
(one line per shard; gzip shards are read transparently).
[`configs/drusilla_model.yaml`](configs/drusilla_model.yaml) is the configuration
of `drusilla_model`, [`configs/vertebrates.yaml`](configs/vertebrates.yaml) that
of the legacy per-clade models. `--resume` continues a run from its newest epoch
checkpoint, including the learning rate it had reached. See
[docs/training.md](docs/training.md) for full details.

Preparing the TFRecords from raw genomes + reference annotations is a
larger data-generation pipeline, described separately in
[docs/data_prep.md](docs/data_prep.md).

---


## License

MIT — see [LICENSE](LICENSE).

## Funding

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation), project number 552910312: "AI-GUSTUS: Eine cloud-native Pipeline
für genaue Genom-Annotation" (AI-GUSTUS: a cloud-native pipeline for accurate
genome annotation).
