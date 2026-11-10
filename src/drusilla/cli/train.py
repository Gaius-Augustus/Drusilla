"""`drusilla train`: train a model from prepared TFRecords.

Manifests are TSV files: one line per shard, last column is the shard path
(any leading columns like species name are ignored)::

    Homo_sapiens\t/data/tfrecords/hs/train.tfrecords
    Mus_musculus\t/data/tfrecords/mm/train.tfrecords.gz

Shards named ``*.tfrecords.gz`` are read GZIP-compressed; both kinds may be
mixed in one manifest.

The config is a YAML with ``data``, ``model``, ``training``, and
``checkpointing`` blocks (see configs/drusilla_model.yaml).

Resuming a run that was stopped by a walltime limit::

    drusilla train ... --out-dir results/models/run_001 --resume

``--resume`` loads the newest ``epoch_NN.weights.h5`` of ``--out-dir``,
continues the epoch numbering, restores the learning rate the run had
reached (read from the ``learning_rate`` column of ``train_log.tsv``; a
ReduceLROnPlateau step is otherwise lost, because Keras restores weights
only), appends to ``train_log.tsv`` instead of truncating it, and seeds the
best-weights checkpoint with the best ``val_loss`` logged so far, so an
early epoch of the resumed job cannot overwrite ``best.weights.h5`` with a
worse model. ``--init-weights`` / ``--initial-epoch`` / ``--lr`` remain
available for manual control and fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--train-manifest", type=Path, required=True,
                    help="TSV of training TFRecords (species<TAB>path).")
    ap.add_argument("--val-manifest", type=Path, default=None,
                    help="TSV of validation TFRecords. Optional; without it "
                         "model.fit runs without validation and the "
                         "val-based callbacks (best weights, early stopping, "
                         "ReduceLROnPlateau) are skipped.")
    ap.add_argument("--config", type=Path, required=True,
                    help="Training config YAML "
                         "(e.g. configs/drusilla_model.yaml).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for checkpoints and logs.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override batch_size from config.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override learning_rate from config (with --resume: "
                         "overrides the learning rate read from train_log.tsv).")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Load these weights before training (resume / fine-tune).")
    ap.add_argument("--initial-epoch", type=int, default=0,
                    help="Epoch index to start numbering at (for resume).")
    ap.add_argument("--resume", action="store_true",
                    help="Continue the run in --out-dir from its newest "
                         "epoch_NN.weights.h5: sets --init-weights and "
                         "--initial-epoch, restores the logged learning rate, "
                         "appends to train_log.tsv. Starts from scratch when "
                         "the directory holds no epoch checkpoint yet.")


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


# ----------------------------------------------------------------------------
# Resume helpers (pure Python, unit-tested without TensorFlow)
# ----------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"^epoch_(\d+)\.weights\.h5$")


def _read_log_column(out_dir: Path, column: str) -> list[float]:
    """Values of ``column`` in ``train_log.tsv`` (rows without the column or
    with a non-numeric value are skipped)."""
    path = out_dir / "train_log.tsv"
    if not path.is_file():
        return []
    values: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, dialect="excel-tab"):
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue
    return values


def _previous_best(out_dir: Path, monitor: str = "val_loss") -> float | None:
    """Best ``monitor`` value logged so far in ``train_log.tsv``, or None.

    A resumed job's best-weights checkpoint starts with no reference value, so
    without this its first epoch would overwrite ``best.weights.h5`` even when
    an earlier epoch was better.
    """
    values = _read_log_column(out_dir, monitor)
    return min(values) if values else None


def _last_logged_lr(out_dir: Path) -> float | None:
    """Learning rate of the last logged epoch (``learning_rate`` column), or None."""
    values = _read_log_column(out_dir, "learning_rate")
    return values[-1] if values else None


def _newest_epoch_checkpoint(out_dir: Path) -> tuple[Path, int] | None:
    """``(path, epoch_number)`` of the highest ``epoch_NN.weights.h5`` in
    ``out_dir``; the epoch number is 1-based as Keras writes it, so it is also
    the ``initial_epoch`` to continue with."""
    best: tuple[int, Path] | None = None
    if not out_dir.is_dir():
        return None
    for p in out_dir.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return (best[1], best[0]) if best else None


def _resolve_resume(args: argparse.Namespace) -> list[str]:
    """Apply ``--resume``: fill init_weights / initial_epoch / lr from
    ``--out-dir``. Returns human-readable notes."""
    notes: list[str] = []
    if not args.resume:
        return notes
    ckpt = _newest_epoch_checkpoint(args.out_dir)
    if ckpt is None:
        notes.append(f"--resume: no epoch_NN.weights.h5 in {args.out_dir}, starting from scratch")
        return notes
    path, epoch = ckpt
    if args.init_weights is None:
        args.init_weights = path
    if args.initial_epoch == 0:
        args.initial_epoch = epoch
    notes.append(f"--resume: weights {path.name}, continuing at epoch {args.initial_epoch}")
    if args.lr is None:
        lr = _last_logged_lr(args.out_dir)
        if lr is not None:
            args.lr = lr
            notes.append(f"--resume: learning rate {lr:g} from train_log.tsv")
        else:
            notes.append("--resume: train_log.tsv has no learning_rate column, "
                         "using the config learning rate (pass --lr to override)")
    return notes


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------

def _build_callbacks(tc: dict, out_dir: Path, has_val: bool, resume: bool = False) -> list:
    import tensorflow as tf

    class _LearningRateLogger(tf.keras.callbacks.Callback):
        """Adds ``learning_rate`` to the epoch logs (and so to train_log.tsv),
        which is what ``--resume`` reads back after a ReduceLROnPlateau step."""
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            lr = getattr(self.model.optimizer, "learning_rate", None)
            try:
                logs["learning_rate"] = float(tf.keras.ops.convert_to_numpy(lr))
            except Exception:    # a schedule object, or no optimizer
                pass

    cbs = [
        _LearningRateLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "epoch_{epoch:02d}.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        ),
        # A resumed run appends, otherwise each walltime-limited job would
        # truncate the log and keep only its own epochs.
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.tsv"), separator="\t",
                                     append=resume),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    if has_val:
        best_so_far = _previous_best(out_dir, "val_loss") if resume else None
        if best_so_far is not None:
            print(f"best val_loss before resume: {best_so_far:.5f}", flush=True)
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.weights.h5"),
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True,
            verbose=1,
            initial_value_threshold=best_so_far,
        ))

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


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for note in _resolve_resume(args):
        print(note, flush=True)

    if args.epochs is not None:
        tc["epochs"] = args.epochs
    if args.batch_size is not None:
        dc["batch_size"] = args.batch_size
    if args.lr is not None:
        tc["learning_rate"] = args.lr

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
        cycle_length=dc.get("cycle_length"),
    ))
    if args.val_manifest is not None:
        val_ds = _pack_y(make_dataset(
            args.val_manifest,
            chunk_len=dc["chunk_len"],
            batch_size=dc["batch_size"],
            shuffle=False,
            repeat=False,
            cycle_length=dc.get("cycle_length"),
        ))
    else:
        val_ds = None
        print("no --val-manifest given: training without validation", flush=True)

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
        # validation_steps caps the per-epoch validation pass (None = all).
        validation_steps=tc.get("validation_steps") if val_ds is not None else None,
        callbacks=_build_callbacks(tc, args.out_dir, has_val=val_ds is not None,
                                   resume=args.initial_epoch > 0),
        initial_epoch=args.initial_epoch,
    )

    model.save_weights(str(args.out_dir / "final.weights.h5"))
    print(f"Training complete. Outputs in {args.out_dir}", flush=True)
    return 0
