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
