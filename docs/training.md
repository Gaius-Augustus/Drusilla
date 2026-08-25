# Training a Drusilla model

This describes the model-training step: given a prepared TFRecord dataset
and a config, run `drusilla train`. See
[data_prep.md](data_prep.md) for how to build the TFRecords.

## Inputs

1. **A training manifest TSV.** One line per shard; last column is the
   `.tfrecords` path. Leading columns (species name, etc.) are ignored:
   ```
   Homo_sapiens        /data/tfrecords/hs.tfrecords
   Mus_musculus        /data/tfrecords/mm.tfrecords
   ...
   ```
2. **A validation manifest** in the same format. Species must not
   overlap with training.
3. **A YAML config** — start from [configs/vertebrates.yaml](../configs/vertebrates.yaml).

## Config schema

```yaml
data:
  chunk_len: 9999             # window length used when writing TFRecords
  batch_size: 400
  shuffle_buffer: 2048

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

## Run

```bash
drusilla train \
  --train-manifest results/tfrecords/train_manifest.tsv \
  --val-manifest   results/tfrecords/val_manifest.tsv \
  --config         configs/vertebrates.yaml \
  --out-dir        results/models/run_001
```

Every epoch produces `epoch_NN.weights.h5` in `--out-dir`. The best-so-far
weights (by `val_loss`) are also written to `best.weights.h5`, along with
a plain-text `train_log.tsv` and a `final.weights.h5` at end of training.

### Overrides without editing the config

```bash
drusilla train --config configs/vertebrates.yaml \
  --train-manifest ... --val-manifest ... --out-dir ... \
  --epochs 100 --batch-size 200 --lr 5e-5
```

### Resume / fine-tune from existing weights

```bash
drusilla train --config configs/vertebrates.yaml \
  --train-manifest ... --val-manifest ... --out-dir ... \
  --init-weights   results/models/run_001/epoch_74.weights.h5 \
  --initial-epoch  75
```

## Loss

Two options via `training.loss_type`:

| `loss_type`         | Term                                          |
|---------------------|-----------------------------------------------|
| `cce`               | Masked, class-weighted categorical cross-entropy. |
| `cce_boundary_f1`   | Above + soft-F1 penalty on START and STOP. |

Padded positions (where `input[..., 5] == 1`) are always excluded from
the loss and metrics.

## Reported metrics

- `accuracy` (masked, per-position argmax)
- `f1_IR / f1_START / f1_E1 / f1_E2 / f1_E0 / f1_STOP`
- `val_*` counterparts on the validation set

## Hardware

- Trained on a single 40 GB A100 GPU with batch size 400 and
  `chunk_len=9999`. Peak GPU memory ≈ 20 GB.
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
`drusilla annotate --weights <path> --config <config>`.
