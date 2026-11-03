# `model_cfg/`

One YAML per released Drusilla model. See [vertebrates.yaml](vertebrates.yaml)
for the schema and inline field reference.

The file names the release; the actual model weights + architecture
config live in the `.tar.gz` archive at `weights_url`. The archive
must extract to a directory named exactly like the model, `<name>/`
(the registry rejects any other top-level directory), containing:

- `model.weights.h5` — Keras weights file (a plain `weights.h5` is
  accepted and renamed on extraction)
- `arch.yaml`  — architecture / data config (parsed with
  `drusilla.model.model.build_model_from_config`)

To publish a new model:

1. Build the archive:
   ```bash
   mkdir vertebrates
   cp path/to/best.weights.h5 vertebrates/model.weights.h5
   cp path/to/arch.yaml       vertebrates/arch.yaml
   tar -czf vertebrates.tar.gz vertebrates/
   ```
2. Upload the tarball to a stable HTTPS location.
3. Add / update a `<name>.yaml` here with the URL, the archive's
   `weights_sha256` (`sha256sum vertebrates.tar.gz`; checked after
   every download) and the `training_run`. Bump `version` to
   invalidate cached copies on client machines: the extracted
   directory carries a `.drusilla_version` marker, and a mismatch
   with the manifest triggers a re-download.

Every `*.yaml` in this directory is bundled into the wheel
(`[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`),
so a new manifest reaches pip and container installs without further
changes.

To use a model manifest outside the bundled ones, set
`DRUSILLA_MODEL_CFG_DIR` to a directory of extra `*.yaml` files (they
override bundled entries with the same `name`).

## Optional manifest fields (Drusilla >= 0.3)

- `annotate_defaults` — inference options that `drusilla annotate --model
  <name>` applies unless the user sets them on the command line. Allowed
  keys (same spelling as the argparse destinations):
  `unstranded_both_strands` (bool), `unstranded_pick`
  (`nonoverlap|longest|both`), `orfs_per_tx` (int >= 0), `flank_bp`,
  `flank_3p_bp` (int >= 0), `flank_keep` (bool). Unknown keys or wrong
  types abort `annotate` with a message. Use this to ship the inference
  rules a model was trained and benchmarked with; the per-clade manifests
  set nothing and so keep the built-in defaults.
- `label_policy` — informational, the label policy of the training data
  (`v1` for the per-clade models, `v2` for the joint model; see
  `docs/data_prep.md`).

A file with another suffix (`joint.yaml.draft`) is ignored by the
registry; use it to prepare a manifest before the weights exist.

## Releasing the joint model

The joint model (`cnn_lstm_joint_run001`, see `docs/training.md`) replaces
the per-clade models as the default. Its manifest is prepared as
[`joint.yaml.draft`](joint.yaml.draft); to release:

1. Pick the epoch: the row with the lowest `val_loss` in
   `results/models/cnn_lstm_joint_run001/train_log.tsv` (the
   `best.weights.h5` of a resumed chain is only reliable from Drusilla
   0.3 on; `epoch_NN.weights.h5` plus the log are authoritative). Confirm
   it on the test benchmarks (tiberius_orf_finder bundle
   `bundles/joint_vs_clade`).
2. Build the archive:
   ```bash
   mkdir joint
   cp results/models/cnn_lstm_joint_run001/epoch_NN.weights.h5 joint/model.weights.h5
   cp configs/joint.yaml joint/arch.yaml
   tar -czf joint.tar.gz joint/
   sha256sum joint.tar.gz
   ```
3. Upload the tarball, then fill the `TODO` fields of `joint.yaml.draft`
   (`weights_url`, `weights_sha256`, `date`, the epoch in the
   `training_run` comment) and rename it to `joint.yaml`. Keep or adjust
   the `annotate_defaults` block (it mirrors the benchmark settings).
4. Make it the default: set `DEFAULT_MODEL = "joint"` in
   `src/drusilla/cli/annotate.py`; update the Models table and the model
   availability note in `README.md` and the model-selection table in
   `docs/annotate.md`; bump `version` in `pyproject.toml` and
   `src/drusilla/__init__.py`.
5. `pytest` — `tests/test_registry.py` checks that every `model_cfg/*.yaml`
   loads and carries a SHA256, `tests/test_cli_smoke.py` validates every
   bundled `annotate_defaults` block.
6. Rebuild and push the container image (the manifest ships inside the
   wheel), then tag the release.

The per-clade manifests stay in place, so `--model vertebrates` and
`--model insects` keep working.
