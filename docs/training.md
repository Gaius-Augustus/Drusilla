# Training a Drusilla model

This describes the model-training step: given a prepared TFRecord dataset
and a config, run `drusilla train`. See
[data_prep.md](data_prep.md) for how to build the TFRecords.

## Inputs

1. **A training manifest TSV.** One line per shard; last column is the
   shard path. Leading columns (species name, etc.) are ignored:
   ```
   Homo_sapiens        /data/tfrecords/hs.tfrecords.gz
   Mus_musculus        /data/tfrecords/mm.tfrecords.gz
   ...
   ```
   Shards named `*.tfrecords.gz` are read GZIP-compressed; plain and
   gzip shards may be mixed.
2. **A validation manifest** in the same format (optional, but without it
   there is no `best.weights.h5`, no ReduceLROnPlateau and no early
   stopping). Species must not overlap with training.
3. **A YAML config** — [configs/drusilla_model.yaml](../configs/drusilla_model.yaml) is the
   configuration of `drusilla_model`.

## Config schema

```yaml
data:
  chunk_len: 9999             # window length used when writing TFRecords
  batch_size: 400
  shuffle_buffer: 8192        # decoded chunks held for shuffling (~120 KB each)
  cycle_length: 0             # shards read side by side: 0 = all, N = at most N,
                              # absent = 32 (the pre-0.3 behaviour)

model:
  type: cnn_lstm              # or cnn_transformer
  conv_filters: 64            # shared CNN stem
  conv_kernel: 9
  conv_layers: 2
  cnn_lstm:
    pool_size: 9              # chunk_len must divide by pool_size
    lstm_units: 200
    lstm_layers: 2
    dropout: 0.0
    head_hidden: 32
  cnn_transformer:
    pool_size: 9
    d_model: 256
    num_heads: 4
    ff_dim: 512
    transformer_layers: 4
    dropout: 0.1
    head_hidden: 32

training:
  loss_type: cce_boundary_f1  # or "cce" for plain masked CCE
  f1_lambda: 1.0              # weight on the START/STOP soft-F1 aux term
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4        # >0 uses AdamW instead of Adam
  epochs: 300
  steps_per_epoch: 2000
  validation_steps: 250       # optional: cap the validation pass per epoch
                              # (batches); absent = the whole validation set
  class_weights: [1, 5, 1, 1, 1, 5]   # [IR, START, E1, E2, E0, STOP]
  warmup_steps: 0             # >0 enables cosine decay w/ linear warmup
  lr_reduce:                  # optional ReduceLROnPlateau
    monitor: val_loss
    patience: 7
    factor: 0.5
    min_lr: 1.0e-6
  early_stopping:             # optional
    monitor: val_loss
    patience: 20

checkpointing:
  save_every_epoch: true
```

`cycle_length` matters for manifests with many shards: the interleave
reads one record from each open shard in turn, so with the old cap of 32
only 32 species at a time reached the shuffle buffer. The run of
`drusilla_model` reads all 425 shards side by side. Note that even then the species mix
drifts within one pass over the data (small shards run dry first), so
training-set metrics of different epochs are not comparable; judge
progress by the validation columns, which are computed on a fixed,
deterministic subset of the validation set.

## Run

```bash
drusilla train \
  --train-manifest results/tfrecords/train_manifest.tsv \
  --val-manifest   results/tfrecords/val_manifest.tsv \
  --config         configs/drusilla_model.yaml \
  --out-dir        results/models/run_001
```

Every epoch produces `epoch_NN.weights.h5` in `--out-dir`. The best-so-far
weights (by `val_loss`) are also written to `best.weights.h5`, along with
`train_log.tsv` (one row per epoch: loss, masked accuracy, per-class F1,
their `val_` counterparts and `learning_rate`) and a `final.weights.h5` at
the end of training.

### Overrides without editing the config

```bash
drusilla train --config configs/drusilla_model.yaml \
  --train-manifest ... --val-manifest ... --out-dir ... \
  --epochs 100 --batch-size 200 --lr 5e-5
```

### Resume after a walltime limit

A full run of 300 epochs takes days; on clusters with a 12 h or 24 h job
limit, submit the same command repeatedly with `--resume` (for example as
a chain of SLURM jobs with `--dependency=afterany:<previous job>`):

```bash
drusilla train --config configs/drusilla_model.yaml \
  --train-manifest ... --val-manifest ... \
  --out-dir results/models/run_001 --resume
```

`--resume` does four things that a plain `--init-weights` restart does
not:

1. loads the newest `epoch_NN.weights.h5` of `--out-dir` and continues the
   epoch numbering at `NN`;
2. restores the learning rate the run had reached, read from the
   `learning_rate` column of `train_log.tsv` (Keras restores weights only,
   so ReduceLROnPlateau steps would otherwise be lost; `--lr` overrides);
3. appends to `train_log.tsv` instead of truncating it;
4. seeds the `best.weights.h5` checkpoint with the best `val_loss` logged
   so far, so an early epoch of the resumed job cannot overwrite a better
   model.

ReduceLROnPlateau's patience counter does restart with every job; a
plateau that spans a job boundary therefore reduces the learning rate a
few epochs later than in an uninterrupted run.

If `--out-dir` holds no epoch checkpoint yet, `--resume` starts from
scratch, so the same command line can be used for the first and every
following job.

### Fine-tune from existing weights

```bash
drusilla train --config configs/drusilla_model.yaml \
  --train-manifest ... --val-manifest ... --out-dir results/models/finetune_001 \
  --init-weights   models/drusilla_model/model.weights.h5 \
  --lr 2e-5
```

`models/drusilla_model/model.weights.h5` is the bundled model in a repository
checkout; in an installed package `drusilla models path drusilla_model`
prints where it is.

`--init-weights` / `--initial-epoch` give manual control over what
`--resume` automates.

## Loss

Two options via `training.loss_type`:

| `loss_type`         | Term                                          |
|---------------------|-----------------------------------------------|
| `cce`               | Masked, class-weighted categorical cross-entropy. |
| `cce_boundary_f1`   | Above + soft-F1 penalty on START and STOP. |

Padded positions (where `input[..., 5] == 1`) are always excluded from
the loss and metrics.

The released models were trained with `cce_boundary_f1`.

## Reported metrics

- `accuracy` (masked, per-position argmax)
- `f1_IR / f1_START / f1_E1 / f1_E2 / f1_E0 / f1_STOP`
- `val_*` counterparts on the validation set
- `learning_rate` (current optimiser value, used by `--resume`)

## The Drusilla model

`drusilla_model` (`configs/drusilla_model.yaml`, training run
`cnn_lstm_joint_run001`) was trained on the pooled TFRecords of 425
species: 68 vertebrates, 88 insects, 32 land plants, 213 fungi, 18
diatoms and 6 green algae, all labelled with label policy v2 and stored
as gzip shards (21.5 GB in total; 13 M chunks of 9999 nt). Validation used
6 held-out species (Ornithorhynchus anatinus, Petromyzon marinus,
Drosophila ananassae, Drosophila virilis, Tenebrio molitor, Populus
trichocarpa), 100k chunks per epoch in deterministic order. One epoch of
2000 steps covers about 6 % of the training chunks and took about 20 min
on one GPU with batch size 400; the run was a chain of 12 h jobs resumed
as described above. It ran for 73 epochs in two jobs: the learning rate
was halved six times to 1.6e-6, the validation loss did not improve after
epoch 42, and the test benchmark of one species per clade scored epoch 73
highest of the epochs tried, so the weights of epoch 73 (the last) ship as
`models/drusilla_model/model.weights.h5` (update steps in
[model_cfg/README.md](../model_cfg/README.md)). The manifest lists the species
([model_cfg/drusilla_model.yaml](../model_cfg/drusilla_model.yaml)); the
TFRecords themselves were produced with the data pipeline of the
[tiberius_orf_finder](https://github.com/Gaius-Augustus/tiberius_orf_finder)
repository on the group's cluster (see [data_prep.md](data_prep.md) for
the per-species steps).

## Hardware

- `drusilla_model`: one GPU (tested on 40 GB A100-class cards) with batch size
  400 and `chunk_len=9999`, 64 GB of host RAM requested (peak about
  17 GB), about 20 min per epoch of 2000 steps.
- Fine-tuning from released weights fits in ~10 GB with batch size 100.

## Output layout

```
results/models/run_001/
├── epoch_01.weights.h5
├── epoch_02.weights.h5
├── ...
├── best.weights.h5
├── final.weights.h5
└── train_log.tsv
```

Any of these `.weights.h5` files can be handed to
`drusilla annotate --weights <path> --config <config>`, or packaged as a
model release (see [model_cfg/README.md](../model_cfg/README.md)).
