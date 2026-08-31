# `model_cfg/`

One YAML per released Drusilla model. See [vertebrates.yaml](vertebrates.yaml)
for the schema and inline field reference.

The file names the release; the actual model weights + architecture
config live in the `.tar.gz` archive at `weights_url`. The archive
should extract to a directory `<name>-v<version>/` containing:

- `weights.h5` — Keras weights file
- `arch.yaml`  — architecture / data config (parsed with
  `drusilla.model.model.build_model_from_config`)

To publish a new model:

1. Build the archive:
   ```bash
   mkdir vertebrates-v1.0
   cp path/to/best.weights.h5 vertebrates-v1.0/weights.h5
   cp path/to/arch.yaml       vertebrates-v1.0/arch.yaml
   tar -czf vertebrates-v1.0.tar.gz vertebrates-v1.0/
   sha256sum vertebrates-v1.0.tar.gz
   ```
2. Upload the tarball to a stable HTTPS location.
3. Add / update a `<name>.yaml` here with the URL and SHA256.

To use a model manifest outside the bundled ones, set
`DRUSILLA_MODEL_CFG_DIR` to a directory of extra `*.yaml` files (they
override bundled entries with the same `name`).
