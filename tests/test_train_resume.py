"""Resume helpers of `drusilla train` (pure Python, no TensorFlow)."""

from __future__ import annotations

import argparse
from pathlib import Path

from drusilla.cli.train import (
    _last_logged_lr,
    _newest_epoch_checkpoint,
    _previous_best,
    _resolve_resume,
)

LOG = (
    "epoch\taccuracy\tloss\tlearning_rate\tval_accuracy\tval_loss\n"
    "0\t0.83\t1.45\t0.0001\t0.90\t1.2574\n"
    "1\t0.89\t1.22\t0.0001\t0.92\t0.9983\n"
    "2\t0.90\t0.89\t5e-05\t0.93\t0.7521\n"
    "3\t0.91\t0.78\t5e-05\t0.94\t0.7700\n"
)


def _run_dir(tmp_path: Path, epochs=(1, 2, 3, 4), log: str | None = LOG) -> Path:
    for n in epochs:
        (tmp_path / f"epoch_{n:02d}.weights.h5").write_bytes(b"x")
    (tmp_path / "best.weights.h5").write_bytes(b"x")
    if log is not None:
        (tmp_path / "train_log.tsv").write_text(log)
    return tmp_path


def test_previous_best_is_the_minimum_val_loss(tmp_path):
    d = _run_dir(tmp_path)
    assert _previous_best(d) == 0.7521
    assert _previous_best(d, "val_accuracy") == 0.90
    assert _previous_best(tmp_path / "missing") is None
    (d / "train_log.tsv").write_text("epoch\tloss\n0\t1.0\n")      # no val column
    assert _previous_best(d) is None


def test_last_logged_lr_and_newest_checkpoint(tmp_path):
    d = _run_dir(tmp_path, epochs=(1, 2, 10, 9))
    assert _last_logged_lr(d) == 5e-05
    path, epoch = _newest_epoch_checkpoint(d)
    assert (path.name, epoch) == ("epoch_10.weights.h5", 10)      # numeric, not lexical
    assert _newest_epoch_checkpoint(tmp_path / "missing") is None
    (d / "train_log.tsv").write_text("epoch\tloss\n0\t1.0\n")
    assert _last_logged_lr(d) is None


def _args(out_dir: Path, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(out_dir=out_dir, resume=True, init_weights=None,
                            initial_epoch=0, lr=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_resolve_resume_fills_weights_epoch_and_lr(tmp_path):
    d = _run_dir(tmp_path)
    ns = _args(d)
    notes = _resolve_resume(ns)
    assert ns.init_weights == d / "epoch_04.weights.h5"
    assert ns.initial_epoch == 4
    assert ns.lr == 5e-05
    assert any("epoch 4" in n for n in notes) and any("5e-05" in n for n in notes)


def test_resolve_resume_respects_explicit_values(tmp_path):
    d = _run_dir(tmp_path)
    ns = _args(d, lr=1e-6, initial_epoch=2, init_weights=d / "epoch_02.weights.h5")
    _resolve_resume(ns)
    assert (ns.lr, ns.initial_epoch, ns.init_weights.name) == (1e-6, 2, "epoch_02.weights.h5")


def test_resolve_resume_without_checkpoints_starts_fresh(tmp_path):
    ns = _args(tmp_path)
    notes = _resolve_resume(ns)
    assert ns.init_weights is None and ns.initial_epoch == 0 and ns.lr is None
    assert notes and "starting from scratch" in notes[0]
    ns = _args(tmp_path, resume=False)
    assert _resolve_resume(ns) == []


def test_resolve_resume_without_lr_column_keeps_config_lr(tmp_path):
    d = _run_dir(tmp_path, log="epoch\tloss\tval_loss\n0\t1.0\t0.9\n")
    ns = _args(d)
    notes = _resolve_resume(ns)
    assert ns.lr is None and ns.initial_epoch == 4
    assert any("no learning_rate column" in n for n in notes)
