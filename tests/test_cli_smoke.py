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
    # the manifest-presettable options parse to None and get the built-in
    # defaults when no manifest block applies: one ORF per transcript,
    # longest orientation, no flanks
    from drusilla.cli.annotate import INFERENCE_DEFAULTS, _apply_inference_defaults

    ns = ap.parse_args(["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa",
                        "--out-dir", "o"])
    assert all(getattr(ns, k) is None for k in INFERENCE_DEFAULTS)
    assert _apply_inference_defaults(ns) == []
    assert ns.orfs_per_tx == 1 and ns.unstranded_pick == "longest"
    assert ns.unstranded_both_strands is False
    assert ns.flank_bp == 0 and ns.flank_3p_bp == 0 and ns.flank_keep is False
    assert ns.fix_strand is False
    # nonoverlap is a valid pick; the boolean options have --no- forms
    ns = ap.parse_args(["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa",
                        "--out-dir", "o", "--unstranded-pick", "nonoverlap",
                        "--no-unstranded-both-strands", "--no-flank-keep"])
    assert ns.unstranded_pick == "nonoverlap"
    assert ns.unstranded_both_strands is False and ns.flank_keep is False


def _manifest_dir(tmp_path, defaults_yaml: str):
    (tmp_path / "custom.yaml").write_text(
        "name: custom\nversion: '0.1'\n"
        "weights_url: 'https://example.com/custom.tar.gz'\n" + defaults_yaml,
        encoding="utf-8",
    )
    return tmp_path


def test_manifest_annotate_defaults_fill_unset_options(tmp_path, monkeypatch):
    """A manifest's annotate_defaults preset the inference options; values
    given on the command line win."""
    from drusilla.cli.annotate import _apply_inference_defaults, _manifest_inference_defaults

    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(_manifest_dir(tmp_path, (
        "annotate_defaults:\n"
        "  unstranded_both_strands: true\n"
        "  unstranded_pick: nonoverlap\n"
        "  orfs_per_tx: 0\n"
        "  flank_bp: 500\n"
        "  flank_3p_bp: 500\n"
        "  flank_keep: true\n"))))
    ap = _build_parser()
    base = ["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa", "--out-dir", "o",
            "--model", "custom"]
    ns = ap.parse_args(base + ["--orfs-per-tx", "2"])
    came_from_manifest = _apply_inference_defaults(ns, _manifest_inference_defaults(ns))
    assert sorted(came_from_manifest) == ["flank_3p_bp", "flank_bp", "flank_keep",
                                          "unstranded_both_strands", "unstranded_pick"]
    assert ns.orfs_per_tx == 2                      # command line wins
    assert ns.unstranded_both_strands is True and ns.unstranded_pick == "nonoverlap"
    assert (ns.flank_bp, ns.flank_3p_bp, ns.flank_keep) == (500, 500, True)
    # a --weights run ignores manifests entirely
    ns = ap.parse_args(["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa",
                        "--out-dir", "o", "--weights", "w.h5", "--config", "c.yaml"])
    assert _manifest_inference_defaults(ns) == {}


def test_manifest_annotate_defaults_are_validated(tmp_path, monkeypatch):
    from drusilla.cli.annotate import _apply_inference_defaults, _manifest_inference_defaults

    ap = _build_parser()
    base = ["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa", "--out-dir", "o",
            "--model", "custom"]
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(_manifest_dir(tmp_path,
                       "annotate_defaults:\n  no_such_option: 1\n")))
    with pytest.raises(SystemExit):
        _manifest_inference_defaults(ap.parse_args(base))
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(_manifest_dir(tmp_path,
                       "annotate_defaults:\n  unstranded_pick: sideways\n")))
    with pytest.raises(SystemExit):
        ns = ap.parse_args(base)
        _apply_inference_defaults(ns, _manifest_inference_defaults(ns))
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(_manifest_dir(tmp_path,
                       "annotate_defaults:\n  orfs_per_tx: -1\n")))
    with pytest.raises(SystemExit):
        ns = ap.parse_args(base)
        _apply_inference_defaults(ns, _manifest_inference_defaults(ns))


def test_bundled_manifests_annotate_defaults_are_valid(tmp_path, monkeypatch):
    """Every bundled manifest's annotate_defaults block (if any) passes the
    same validation the CLI applies."""
    from drusilla import registry
    from drusilla.cli.annotate import _apply_inference_defaults, _manifest_inference_defaults

    ap = _build_parser()
    for name in registry.list_manifests():
        ns = ap.parse_args(["annotate", "--stringtie-gtf", "st.gtf", "--genome", "g.fa",
                            "--out-dir", "o", "--model", name])
        _apply_inference_defaults(ns, _manifest_inference_defaults(ns))
        assert ns.unstranded_pick in ("nonoverlap", "longest", "both")


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
