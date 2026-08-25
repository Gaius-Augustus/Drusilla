"""Smoke tests for the CLI (no TF, no downloads)."""

from __future__ import annotations

import pytest

from drusilla.cli import _build_parser, main


def test_parser_builds():
    ap = _build_parser()
    assert ap.prog == "drusilla"


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_annotate_help():
    with pytest.raises(SystemExit) as exc:
        main(["annotate", "--help"])
    assert exc.value.code == 0


def test_train_help():
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0


def test_models_help():
    with pytest.raises(SystemExit) as exc:
        main(["models", "--help"])
    assert exc.value.code == 0


def test_models_list_runs(capsys, tmp_path, monkeypatch):
    """`drusilla models list` shouldn't require TF or network."""
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    rc = main(["models", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vertebrates" in out


def test_missing_required_args_fails():
    with pytest.raises(SystemExit):
        main(["annotate"])
