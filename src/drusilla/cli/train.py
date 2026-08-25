"""`drusilla train`: train a model from prepared TFRecords.

Manifests are TSV files: one line per shard, last column is the shard path
(any leading columns like species name are ignored)::

    Homo_sapiens\t/data/tfrecords/hs/train.tfrecords
    Mus_musculus\t/data/tfrecords/mm/train.tfrecords

The config is a YAML with ``data``, ``model``, ``training``, and
``checkpointing`` blocks (see configs/vertebrates.yaml).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--train-manifest", type=Path, required=True,
                    help="TSV of training TFRecords (species<TAB>path).")
    ap.add_argument("--val-manifest", type=Path, required=True,
                    help="TSV of validation TFRecords.")
    ap.add_argument("--config", type=Path, required=True,
                    help="Training config YAML "
                         "(e.g. configs/vertebrates.yaml).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for checkpoints and logs.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override batch_size from config.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override learning_rate from config.")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Load these weights before training (resume / fine-tune).")
    ap.add_argument("--initial-epoch", type=int, default=0,
                    help="Epoch index to start numbering at (for resume).")


def _pack_y(ds):
    """Reformat (x, y, pad_mask) -> (x, packed_y) for model.fit().

    packed_y has shape [B, L, 7]: first 6 channels = one-hot labels,
    last channel = float32 pad flag (1.0 = padded position).
    """
    import tensorflow as tf
    def _pack(x, y, pad_mask):
        pad_float = tf.cast(pad_mask[..., tf.newaxis], tf.float32)
        packed = tf.concat([y, pad_float], axis=-1)
        packed.set_shape([None, None, 7])
        return x, packed
    return ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)


def _build_optimizer(tc: dict, model_type: str):
    import tensorflow as tf

    lr     = tc["learning_rate"]
    wd     = tc.get("weight_decay", 0.0)
    warmup = tc.get("warmup_steps", 0)

    if warmup > 0:
        total_steps = tc["epochs"] * tc["steps_per_epoch"]
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=total_steps - warmup,
            warmup_target=lr,
            warmup_steps=warmup,
        )
    else:
        lr_schedule = lr

    if wd > 0.0:
        return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=wd)
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)


def _build_callbacks(tc: dict, out_dir: Path) -> list:
    import tensorflow as tf

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "epoch_{epoch:02d}.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.weights.h5"),
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.tsv"), separator="\t"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    es_cfg = tc.get("early_stopping", {})
    if es_cfg:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=es_cfg.get("patience", 20),
            restore_best_weights=True,
            verbose=1,
        ))

    lrr_cfg = tc.get("lr_reduce", {})
    if lrr_cfg:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lrr_cfg.get("monitor", "val_loss"),
            patience=lrr_cfg.get("patience", 7),
            factor=lrr_cfg.get("factor", 0.5),
            min_lr=lrr_cfg.get("min_lr", 1e-6),
            verbose=1,
        ))

    return cbs


def run(args: argparse.Namespace) -> int:
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]

    if args.epochs is not None:
        tc["epochs"] = args.epochs
    if args.batch_size is not None:
        dc["batch_size"] = args.batch_size
    if args.lr is not None:
        tc["learning_rate"] = args.lr

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from ..data.dataset import make_dataset
    from ..model.model import build_model_from_config
    from ..model.loss import (
        MaskedCategoricalCrossentropy, MaskedCCEPlusBoundaryF1,
        MaskedAccuracy, all_class_f1_metrics,
    )

    print(f"TF version: {tf.__version__}", flush=True)

    train_ds = _pack_y(make_dataset(
        args.train_manifest,
        chunk_len=dc["chunk_len"],
        batch_size=dc["batch_size"],
        shuffle=True,
        shuffle_buffer=dc["shuffle_buffer"],
        repeat=True,
    ))
    val_ds = _pack_y(make_dataset(
        args.val_manifest,
        chunk_len=dc["chunk_len"],
        batch_size=dc["batch_size"],
        shuffle=False,
        repeat=False,
    ))

    model = build_model_from_config(cfg, chunk_len=dc["chunk_len"])
    print(f"Model type: {mc['type']}", flush=True)
    model.summary()

    optimizer = _build_optimizer(tc, mc["type"])
    loss_type = tc.get("loss_type", "cce")
    if loss_type == "cce_boundary_f1":
        loss_fn = MaskedCCEPlusBoundaryF1(
            class_weights=tc["class_weights"],
            f1_lambda=tc.get("f1_lambda", 1.0),
        )
    else:
        loss_fn = MaskedCategoricalCrossentropy(class_weights=tc["class_weights"])
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[MaskedAccuracy(name="accuracy")] + all_class_f1_metrics(),
    )

    if args.init_weights is not None:
        print(f"loading weights from {args.init_weights}", flush=True)
        model.load_weights(str(args.init_weights))

    model.fit(
        train_ds,
        epochs=tc["epochs"],
        steps_per_epoch=tc["steps_per_epoch"],
        validation_data=val_ds,
        callbacks=_build_callbacks(tc, args.out_dir),
        initial_epoch=args.initial_epoch,
    )

    model.save_weights(str(args.out_dir / "final.weights.h5"))
    print(f"Training complete. Outputs in {args.out_dir}", flush=True)
    return 0
