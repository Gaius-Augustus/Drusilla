# `model_cfg/`

One YAML per registered Drusilla model. The registry reads every `*.yaml`
in this directory (files with another suffix, such as `*.draft`, are
ignored); `DRUSILLA_MODEL_CFG_DIR` adds a directory of extra manifests that
override bundled entries with the same `name`. All manifests ship in the
wheel (`[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`).

A manifest names its weights in one of two ways.

## Bundled model (`weights_path`): `drusilla_model`

[drusilla_model.yaml](drusilla_model.yaml) is the released Drusilla model.
Its weights and architecture config are files of the repository and of the
wheel:

- `models/drusilla_model/model.weights.h5`: Keras weights (28 MB), epoch 73
  of training run `cnn_lstm_joint_run001`
- `configs/drusilla_model.yaml`: architecture / data config (parsed with
  `drusilla.model.model.build_model_from_config`)

`weights_path` and `arch_path` are relative to the directory that holds
`model_cfg/` (the repository root in a checkout, `drusilla/_bundled/` in an
installed wheel; the manifest's own directory is tried second). Nothing is
downloaded or cached; `weights_sha256` is the SHA256 of the weights file and
is verified every time the model is resolved.

To ship new weights for `drusilla_model`:

1. Pick the epoch. For a run with `train_log.tsv`, read `val_loss`,
   `val_f1_START` and `val_f1_STOP` (`epoch_NN.weights.h5` plus the log are
   authoritative; the `best.weights.h5` of a resumed chain is only reliable
   from Drusilla 0.3 on) and confirm the choice on the test benchmarks
   (tiberius_orf_finder bundle `bundles/joint_vs_clade`, `compare_epochs.sh`).
   Epoch 73 of `cnn_lstm_joint_run001` was chosen on 2026-11-04: the learning
   rate had been halved six times to 1.6e-6, the validation loss had been
   flat since epoch 42, and epoch 73 scored best of epochs 42, 59, 61 and 73
   on one test species per clade at gene, transcript and exon level.
2. Copy the weights and, if the architecture changed, the config:
   ```bash
   cp path/to/epoch_NN.weights.h5 models/drusilla_model/model.weights.h5
   cp path/to/run_config.yaml     configs/drusilla_model.yaml
   sha256sum models/drusilla_model/model.weights.h5
   ```
3. Put the new SHA256 into `weights_sha256`, update `training_run`, `date`
   and the comment, bump `version` (shown by `drusilla models list`), keep
   or adjust `annotate_defaults`.
4. `pytest`: `tests/test_registry.py` resolves the bundled model and checks
   its SHA256, `tests/test_cli_smoke.py` validates every `annotate_defaults`
   block and the default model.
5. Rebuild and push the container image, tag the release. The weights are a
   plain git object (no LFS), so every replacement adds its size to the
   history; replace them for real releases only.

## Downloaded model (`weights_url`): legacy `insects`

[insects.yaml](insects.yaml) is the insect model of Drusilla 0.2, kept for
reproducibility (`--model insects`); it also holds the inline field
reference.
The weights live in a `.tar.gz` archive at `weights_url`, which must extract
to a directory named exactly like the model, `<name>/`, containing:

- `model.weights.h5`: Keras weights file (a plain `weights.h5` is accepted
  and renamed on extraction)
- `arch.yaml`: architecture / data config

The archive is downloaded into the cache on first use (`DRUSILLA_CACHE_DIR`,
default `~/.cache/drusilla/models/`), checked against `weights_sha256`, and
the extracted directory carries a `.drusilla_version` marker; bumping
`version` in the manifest triggers a re-download. To publish such a model:

```bash
mkdir <name>
cp path/to/best.weights.h5 <name>/model.weights.h5
cp path/to/arch.yaml       <name>/arch.yaml
tar -czf <name>.tar.gz <name>/
sha256sum <name>.tar.gz
```

then upload the tarball to a stable HTTPS location and add `<name>.yaml`
here with `weights_url`, `weights_sha256` and `training_run`.

## Optional manifest fields (Drusilla >= 0.3)

- `annotate_defaults`: inference options that `drusilla annotate` applies
  for this model unless the user sets them on the command line. Allowed
  keys (same spelling as the argparse destinations):
  `unstranded_both_strands` (bool), `unstranded_pick`
  (`nonoverlap|longest|both`), `orfs_per_tx` (int >= 0), `flank_bp`,
  `flank_3p_bp` (int >= 0), `flank_keep` (bool). Unknown keys or wrong
  types abort `annotate` with a message. `drusilla_model` presets the
  inference rules it was trained and benchmarked with; the `insects`
  manifest sets nothing and so keeps the built-in defaults.
- `label_policy`: informational, the label policy of the training data
  (`v1` for `insects`, `v2` for `drusilla_model`; see
  `docs/data_prep.md`).
