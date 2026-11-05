"""Model registry: per-model YAML manifests with bundled weights or tar.gz archives.

Each model is described by a ``model_cfg/<name>.yaml`` manifest (bundled
with the package, plus optional overrides, see ``DRUSILLA_MODEL_CFG_DIR``).
A manifest names its weights in one of two ways:

* ``weights_path`` (bundled model): a Keras ``*.weights.h5`` file that ships
  inside the package / repository, with ``arch_path`` pointing at its
  architecture config (parsed by ``drusilla.model.model.build_model_from_config``).
  Relative paths are resolved against the directory that holds ``model_cfg/``
  (the repository root in a checkout, ``drusilla/_bundled/`` in an installed
  wheel), then against the manifest's own directory. Nothing is downloaded
  or cached; ``weights_sha256`` is the SHA256 of the weights file and is
  verified on every resolve. The released Drusilla model (``drusilla_model``)
  ships this way.
* ``weights_url`` (downloaded model): a ``.tar.gz`` archive that extracts to
  a directory ``<name>/`` containing ``model.weights.h5`` (a plain
  ``weights.h5`` is renamed on extraction, see ``_resolve_weights_path``)
  and ``arch.yaml``. The archive is downloaded into the cache on first use,
  checked against ``weights_sha256``, and the extracted directory is
  stamped with the manifest version (``.drusilla_version`` marker), so
  bumping ``version`` triggers a re-download, as does
  ``drusilla models download NAME --force``. The legacy per-clade models
  (``vertebrates``, ``insects``) ship this way.

Runtime overrides:

* ``DRUSILLA_MODEL_CFG_DIR``: extra directory of ``*.yaml`` files that
  shadow / add to the bundled manifests.
* ``DRUSILLA_CACHE_DIR``: override the cache directory for downloaded
  models (default: ``$XDG_CACHE_HOME/drusilla`` or ``~/.cache/drusilla``).
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_CHUNK = 1 << 20  # 1 MiB stream chunk


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class RegistryError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Bundled + user model_cfg directories
# --------------------------------------------------------------------------

def _bundled_model_cfg_dirs() -> list[Path]:
    """Directories that may hold bundled ``model_cfg/*.yaml`` files.

    Editable-install fallback climbs up from the source tree; wheel
    installs place the files under ``drusilla/_bundled/model_cfg/``.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "_bundled" / "model_cfg",   # wheel install
        here.parents[2] / "model_cfg",            # editable install (repo root)
        here.parents[3] / "model_cfg",            # editable install (parent)
    ]
    return [p for p in candidates if p.is_dir()]


def _user_model_cfg_dirs() -> list[Path]:
    env = os.environ.get("DRUSILLA_MODEL_CFG_DIR")
    if not env:
        return []
    paths = []
    for chunk in env.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            p = Path(chunk).expanduser()
            if p.is_dir():
                paths.append(p)
    return paths


def _all_model_cfg_files() -> dict[str, Path]:
    """Return ``{name: yaml_path}``, later directories overriding earlier ones.

    The lookup order is: bundled -> user override (env var). This lets
    a developer point ``DRUSILLA_MODEL_CFG_DIR`` at their own directory
    to test a new release without editing the package.
    """
    out: dict[str, Path] = {}
    for d in _bundled_model_cfg_dirs():
        for p in sorted(d.glob("*.yaml")):
            out[p.stem] = p
    for d in _user_model_cfg_dirs():
        for p in sorted(d.glob("*.yaml")):
            out[p.stem] = p
    return out


# --------------------------------------------------------------------------
# Cache location
# --------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the Drusilla models cache directory (created on demand)."""
    override = os.environ.get("DRUSILLA_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) / "drusilla" if xdg else Path.home() / ".cache" / "drusilla"
    models = base / "models"
    models.mkdir(parents=True, exist_ok=True)
    return models


# --------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------

_REQUIRED_MANIFEST_KEYS = ("name", "version")
_WEIGHTS_SOURCE_KEYS = ("weights_url", "weights_path")


@dataclass(frozen=True)
class ModelManifest:
    name: str
    version: str
    manifest_path: Path
    data: dict[str, Any]
    weights_url: str | None = None     # downloaded model: .tar.gz archive
    weights_path: str | None = None    # bundled model: weights file in the package
    arch_path: str | None = None       # bundled model: architecture config

    @property
    def bundled(self) -> bool:
        """True when the weights ship with the package (``weights_path``)."""
        return self.weights_path is not None

    @property
    def archive_filename(self) -> str:
        """The archive filename, taken from the URL (downloaded models only)."""
        if self.weights_url is None:
            raise RegistryError(f"model {self.name!r} is bundled and has no archive")
        return self.weights_url.rsplit("/", 1)[-1] or f"{self.name}-v{self.version}.tar.gz"

    @property
    def extract_dirname(self) -> str:
        """The subdirectory name inside the cache after extraction.

        Uses the plain model name (no version suffix) so the on-disk
        layout matches the archive's own top-level directory. Version
        invalidation is handled via a ``.drusilla_version`` marker file
        written into the extracted dir; see ``resolve_model``.
        """
        return self.name


@dataclass(frozen=True)
class ResolvedModel:
    """Local, on-disk paths of a resolved model (bundled, or downloaded + extracted)."""
    name: str
    version: str
    weights_path: Path
    arch_config_path: Path
    manifest: ModelManifest


def _load_manifest_file(path: Path) -> ModelManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for k in _REQUIRED_MANIFEST_KEYS:
        if not data.get(k):
            raise RegistryError(
                f"model manifest {path} is missing required field {k!r}"
            )
    sources = [k for k in _WEIGHTS_SOURCE_KEYS if data.get(k)]
    if len(sources) != 1:
        raise RegistryError(
            f"model manifest {path} must set exactly one of "
            f"{' / '.join(_WEIGHTS_SOURCE_KEYS)} (found: {', '.join(sources) or 'none'})"
        )
    name = str(data["name"])
    if name != path.stem:
        raise RegistryError(
            f"model manifest {path}: name field {name!r} does not match "
            f"filename stem {path.stem!r}"
        )
    return ModelManifest(
        name=name,
        version=str(data["version"]),
        manifest_path=path,
        data=data,
        weights_url=str(data["weights_url"]) if data.get("weights_url") else None,
        weights_path=str(data["weights_path"]) if data.get("weights_path") else None,
        arch_path=str(data["arch_path"]) if data.get("arch_path") else None,
    )


def load_manifest(name: str) -> ModelManifest:
    """Load the manifest for ``name`` from the bundled + user model_cfg dirs."""
    files = _all_model_cfg_files()
    if name not in files:
        available = ", ".join(sorted(files)) or "(none)"
        raise RegistryError(
            f"unknown model {name!r}. Available: {available}"
        )
    return _load_manifest_file(files[name])


def list_manifests() -> dict[str, ModelManifest]:
    """Return ``{name: ModelManifest}`` for every discovered model."""
    return {name: _load_manifest_file(p)
            for name, p in _all_model_cfg_files().items()}


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------

def _sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sha256(archive: Path, expected: str | None) -> None:
    """Delete ``archive`` and raise if its sha256 is not ``expected``.

    Manifests without ``weights_sha256`` are not checked.
    """
    if not expected:
        return
    got = _sha256_of(archive)
    if got != str(expected).strip().lower():
        archive.unlink()
        raise RegistryError(
            f"sha256 of {archive.name} is {got}, the manifest expects {expected}"
        )


# --------------------------------------------------------------------------
# Bundled models (weights_path)
# --------------------------------------------------------------------------

def _bundled_file(mf: ModelManifest, rel: str, what: str) -> Path:
    """Resolve a manifest path: absolute as is, else relative to the directory
    holding ``model_cfg/`` (repo root / ``_bundled``), then to the manifest dir."""
    p = Path(rel).expanduser()
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [mf.manifest_path.parent.parent / p, mf.manifest_path.parent / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise RegistryError(f"model {mf.name!r}: {what} {rel!r} not found (tried {tried})")


def _resolve_bundled(mf: ModelManifest) -> ResolvedModel:
    """Locate and verify the files of a bundled model. Never deletes anything."""
    assert mf.weights_path is not None
    weights = _bundled_file(mf, mf.weights_path, "weights_path")
    if mf.arch_path:
        arch = _bundled_file(mf, mf.arch_path, "arch_path")
    else:
        arch = weights.parent / "arch.yaml"
        if not arch.is_file():
            raise RegistryError(
                f"model {mf.name!r}: set arch_path or place arch.yaml next to {weights}"
            )
    expected = mf.data.get("weights_sha256")
    if expected:
        got = _sha256_of(weights)
        if got != str(expected).strip().lower():
            raise RegistryError(
                f"sha256 of {weights} is {got}, the manifest expects {expected}"
            )
    return ResolvedModel(
        name=mf.name,
        version=mf.version,
        weights_path=weights,
        arch_config_path=arch,
        manifest=mf,
    )


# --------------------------------------------------------------------------
# Download / extract
# --------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            # resp.length is an HTTP-only convenience; file:// responses
            # don't have it. Fall back to Content-Length or None.
            total: int | None = getattr(resp, "length", None)
            if total is None:
                try:
                    total = int(resp.headers.get("Content-Length") or 0) or None
                except (AttributeError, TypeError, ValueError):
                    total = None
            written = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100.0 * written / total
                        print(
                            f"  downloading {dest.name}: "
                            f"{written / 1e6:.1f} / {total / 1e6:.1f} MB "
                            f"({pct:.1f}%)",
                            end="\r", flush=True, file=sys.stderr,
                        )
        if total:
            print("", file=sys.stderr)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _safe_extract(archive: Path, dest_parent: Path, expected_dirname: str) -> Path:
    """Extract ``archive`` under ``dest_parent`` and return the extracted directory.

    Guards against path-traversal ('tar-slip') by rejecting members whose
    resolved path escapes ``dest_parent``. Also refuses archives that
    don't produce the expected top-level directory ``expected_dirname``
    (the model name, see ``ModelManifest.extract_dirname``).
    """
    dest_parent = dest_parent.resolve()
    with tarfile.open(archive, mode="r:*") as tf:
        for member in tf.getmembers():
            member_path = (dest_parent / member.name).resolve()
            try:
                member_path.relative_to(dest_parent)
            except ValueError:
                raise RegistryError(
                    f"archive {archive} contains unsafe path {member.name!r}"
                )
        # filter='data' (Python 3.12+) blocks unsafe members (device
        # files, absolute paths, symlinks escaping the extract root).
        # Our loop above already guards against path traversal but this
        # is a second belt: safer AND silences the 3.14 default warning.
        try:
            tf.extractall(dest_parent, filter="data")
        except TypeError:
            tf.extractall(dest_parent)
    extracted = dest_parent / expected_dirname
    if not extracted.is_dir():
        raise RegistryError(
            f"archive {archive} did not extract to expected "
            f"directory {expected_dirname!r} under {dest_parent}"
        )
    return extracted


_CANONICAL_WEIGHTS_NAME = "model.weights.h5"


def _resolve_weights_path(extract_dir: Path) -> Path | None:
    """Return the path to the Keras weights file inside ``extract_dir``.

    Keras 3 chooses the H5 loader based on the filename suffix:
      * ``*.weights.h5`` -> new hierarchical loader
      * plain ``*.h5``   -> legacy loader
    Files saved by ``ModelCheckpoint(save_weights_only=True)`` on a name
    ending in ``.weights.h5`` use the *new* format on disk. If such a
    file is later renamed to plain ``weights.h5``, the legacy loader
    picks it up and fails with "0 saved layers".

    To insulate publishers from this pitfall we accept either naming in
    the archive and normalise to ``model.weights.h5`` on extract.
    """
    canonical = extract_dir / _CANONICAL_WEIGHTS_NAME
    if canonical.exists():
        return canonical
    # Any user-shipped .weights.h5 file is already in the right format.
    for candidate in sorted(extract_dir.glob("*.weights.h5")):
        return candidate
    # Rename plain weights.h5 -> model.weights.h5 so Keras 3 picks the
    # new-format loader.
    legacy = extract_dir / "weights.h5"
    if legacy.exists():
        legacy.rename(canonical)
        return canonical
    return None


def _expected_files(extract_dir: Path) -> tuple[Path | None, Path]:
    return _resolve_weights_path(extract_dir), extract_dir / "arch.yaml"


_VERSION_MARKER = ".drusilla_version"


def _read_cached_version(extract_dir: Path) -> str | None:
    marker = extract_dir / _VERSION_MARKER
    if not marker.exists():
        return None
    return marker.read_text(encoding="utf-8").strip()


def _write_cached_version(extract_dir: Path, version: str) -> None:
    (extract_dir / _VERSION_MARKER).write_text(version, encoding="utf-8")


def resolve_model(name: str, *, force: bool = False) -> ResolvedModel:
    """Return a :class:`ResolvedModel` for ``name``, downloading if needed.

    A bundled model (``weights_path``) is read in place after its SHA256
    check; ``force`` has no effect on it. For a downloaded model: if the
    extracted directory already exists in the cache with both expected
    files and a ``.drusilla_version`` marker equal to the manifest version,
    it is reused. With ``force=True`` the archive is re-downloaded and
    re-extracted from scratch. To force a re-download across a manifest
    change without touching ``force``, bump the ``version`` field in the
    manifest: the marker then no longer matches.
    """
    mf = load_manifest(name)
    if mf.bundled:
        return _resolve_bundled(mf)

    extract_dir = cache_dir() / mf.extract_dirname
    weights_path, arch_path = _expected_files(extract_dir)

    cached_version = _read_cached_version(extract_dir)
    if (
        not force
        and weights_path is not None
        and arch_path.exists()
        and cached_version == mf.version
    ):
        return ResolvedModel(
            name=mf.name,
            version=mf.version,
            weights_path=weights_path,
            arch_config_path=arch_path,
            manifest=mf,
        )

    # Cache miss OR version mismatch (manifest bumped since last download).
    if cached_version is not None and cached_version != mf.version:
        print(
            f"[registry] {mf.name}: cached version {cached_version} != "
            f"manifest version {mf.version}; re-downloading.",
            file=sys.stderr,
        )
        # Drop the stale archive too so we don't re-extract it.
        stale_archive = cache_dir() / mf.archive_filename
        if stale_archive.exists():
            stale_archive.unlink()

    # (Re-)fetch: download the archive to a sibling ``.part`` file
    # (handled inside ``_download``), then extract in place. Any
    # half-extracted state from a previous failed run is nuked so the
    # next invocation always starts clean.
    archive_path = cache_dir() / mf.archive_filename
    if force and archive_path.exists():
        archive_path.unlink()
    if not archive_path.exists():
        print(f"Downloading {mf.name} v{mf.version} from {mf.weights_url}",
              file=sys.stderr)
        _download(mf.weights_url, archive_path)
        _check_sha256(archive_path, mf.data.get("weights_sha256"))

    if extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)
    try:
        _safe_extract(archive_path, cache_dir(), mf.extract_dirname)
    except (tarfile.TarError, RegistryError):
        # Corrupted / truncated archive: delete so the next invocation
        # triggers a fresh download instead of looping on a bad file.
        if archive_path.exists():
            archive_path.unlink()
        raise

    # Re-resolve after extract: _resolve_weights_path may rename
    # weights.h5 -> model.weights.h5 for Keras 3 compatibility.
    weights_path, arch_path = _expected_files(extract_dir)
    if weights_path is None or not arch_path.exists():
        raise RegistryError(
            f"model {name!r} archive must contain a *.weights.h5 (or "
            f"weights.h5) file and arch.yaml under {extract_dir}"
        )

    # Stamp the extract dir with the manifest version so a later
    # version bump gets detected automatically on the next resolve.
    _write_cached_version(extract_dir, mf.version)

    return ResolvedModel(
        name=mf.name,
        version=mf.version,
        weights_path=weights_path,
        arch_config_path=arch_path,
        manifest=mf,
    )


def clear(name: str) -> list[Path]:
    """Delete cached artifacts for ``name``. Returns removed paths.

    Bundled models have no cache; nothing is removed for them.
    """
    import shutil
    mf = load_manifest(name)
    removed: list[Path] = []
    if mf.bundled:
        return removed
    extract_dir = cache_dir() / mf.extract_dirname
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
        removed.append(extract_dir)
    archive_path = cache_dir() / mf.archive_filename
    if archive_path.exists():
        archive_path.unlink()
        removed.append(archive_path)
    return removed


def local_status(name: str) -> dict[str, Any]:
    """Describe the local state of ``name``.

    ``bundled`` tells which kind of model it is. For a bundled model
    ``cached`` is True when its files are present and the SHA256 matches
    (``error`` holds the reason otherwise). For a downloaded model
    ``cached`` is True only when :func:`resolve_model` would reuse the
    extracted directory without downloading: both expected files exist
    and the ``.drusilla_version`` marker equals the manifest version.
    ``files_present`` ignores the marker (stale cache of an older
    release); ``cached_version`` is the marker value or ``None``.
    """
    mf = load_manifest(name)
    if mf.bundled:
        try:
            r = _resolve_bundled(mf)
        except RegistryError as e:
            return {
                "name": name, "version": mf.version, "bundled": True,
                "cached": False, "files_present": False, "cached_version": None,
                "extract_dir": None, "weights_path": None, "arch_config_path": None,
                "error": str(e),
            }
        return {
            "name": name, "version": mf.version, "bundled": True,
            "cached": True, "files_present": True, "cached_version": mf.version,
            "extract_dir": None,
            "weights_path": str(r.weights_path),
            "arch_config_path": str(r.arch_config_path),
        }
    extract_dir = cache_dir() / mf.extract_dirname
    weights_path, arch_path = _expected_files(extract_dir)
    files_present = weights_path is not None and arch_path.exists()
    cached_version = _read_cached_version(extract_dir) if files_present else None
    cached = files_present and cached_version == mf.version
    return {
        "name": name,
        "version": mf.version,
        "bundled": False,
        "cached": cached,
        "files_present": files_present,
        "cached_version": cached_version,
        "extract_dir": str(extract_dir) if files_present else None,
        "weights_path": str(weights_path) if files_present else None,
        "arch_config_path": str(arch_path) if files_present else None,
    }
