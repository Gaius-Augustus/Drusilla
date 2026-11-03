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


def test_strand_fix_help():
    with pytest.raises(SystemExit) as exc:
        main(["strand-fix-bam", "--help"])
    assert exc.value.code == 0


def test_annotate_transcript_handling_flags_parse():
    ap = _build_parser()
    ns = ap.parse_args([
        "annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa", "--out-dir", "o",
        "--unstranded-both-strands", "--unstranded-pick", "both",
        "--flank-bp", "500", "--flank-3p-bp", "300", "--flank-keep",
        "--flank-min-tx-frac", "0.6", "--orfs-per-tx", "0",
    ])
    assert ns.unstranded_both_strands is True and ns.unstranded_pick == "both"
    assert (ns.flank_bp, ns.flank_3p_bp, ns.flank_keep) == (500, 300, True)
    assert ns.flank_min_tx_frac == 0.6 and ns.orfs_per_tx == 0
    # defaults: one ORF per transcript, longest orientation, no flanks
    ns = ap.parse_args(["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa",
                        "--out-dir", "o"])
    assert ns.orfs_per_tx == 1 and ns.unstranded_pick == "longest"
    assert ns.flank_3p_bp == 0 and ns.flank_keep is False and ns.fix_strand is False


def test_annotate_option_consistency():
    from drusilla.cli.annotate import _check_option_consistency

    ap = _build_parser()
    base = ["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa", "--out-dir", "o"]
    _check_option_consistency(ap.parse_args(base + ["--flank-bp", "100", "--flank-keep"]))
    for bad in (["--flank-keep"],
                ["--flank-bp", "100", "--flank-keep", "--flank-clip"],
                ["--prefix-pad-n", "50", "--flank-3p-bp", "100"],
                ["--fix-strand"],
                ["--orfs-per-tx", "-1"]):
        with pytest.raises(SystemExit):
            _check_option_consistency(ap.parse_args(base + bad))


def test_models_list_shows_all_bundled_models(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "vertebrates" in out and "insects" in out


def test_strand_run_regex_default_matches_module_default():
    """annotate --fix-strand uses the same run-id regex as strand-fix-bam, and
    an explicit 'none' reaches fix_strand unchanged (one-library mode)."""
    from drusilla.data.strand_fix_bam import DEFAULT_RUN_REGEX

    ap = _build_parser()
    base = ["annotate", "--bam", "a.bam", "--genome", "g.fa", "--out-dir", "o"]
    assert ap.parse_args(base).strand_run_regex == DEFAULT_RUN_REGEX
    assert ap.parse_args(base + ["--strand-run-regex", "none"]).strand_run_regex == "none"
