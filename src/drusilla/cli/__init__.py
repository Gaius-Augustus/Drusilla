"""Drusilla command-line entry point.

Usage:
    drusilla annotate ...
    drusilla train ...
    drusilla models list|download|rm|path <NAME>
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="drusilla",
        description="Deep-learning ORF annotator for RNA-seq assembled transcripts.",
    )
    ap.add_argument("--version", action="version", version=f"drusilla {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    from . import annotate as _annotate
    from . import train as _train
    from . import models as _models

    p_ann = sub.add_parser("annotate", help="Annotate a transcriptome with predicted ORFs.")
    _annotate.add_args(p_ann)
    p_ann.set_defaults(_func=_annotate.run)

    p_tr = sub.add_parser("train", help="Train a model from prepared TFRecords.")
    _train.add_args(p_tr)
    p_tr.set_defaults(_func=_train.run)

    p_md = sub.add_parser("models", help="Inspect and manage local model cache.")
    _models.add_args(p_md)
    p_md.set_defaults(_func=_models.run)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return int(args._func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
