"""Model registry — Tiberius-style per-model YAML manifests + tar.gz archives.

Each released model is described by a `model_cfg/<name>.yaml` file
(bundled with the package, plus optional overrides — see
``DRUSILLA_MODEL_CFG_DIR``). The manifest's ``weights_url`` points at a
``.tar.gz`` archive whose contents extract to a directory
``<name>-v<version>/`` containing:

    weights.h5   Keras weights file
    arch.yaml    architecture / data config (parsed by
                 ``drusilla.model.model.build_model_from_config``)

Resolution flow for ``drusilla annotate --model <name>``:

1. Load the manifest from ``model_cfg/<name>.yaml``.
2. If the extracted directory is already present in the cache with all
   expected files, use it directly (no network).
3. Otherwise download the archive, verify its SHA256, extract into the
   cache, and return the paths.

Runtime overrides:

* ``DRUSILLA_MODEL_CFG_DIR`` — extra directory of ``*.yaml`` files that
  shadow / add to the bundled manifests.
* ``DRUSILLA_CACHE_DIR`` — override the cache directory
  (default: ``$XDG_CACHE_HOME/drusilla`` or ``~/.cache/drusilla``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

    The lookup order is: bundled → user override (env var). This lets
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

_REQUIRED_MANIFEST_KEYS = ("name", "version", "weights_url", "weights_sha256")


@dataclass(frozen=True)
class ModelManifest:
    name: str
    version: str
    weights_url: str
    weights_sha256: str
    manifest_path: Path
    data: dict[str, Any]

    @property
    def archive_filename(self) -> str:
        """The archive filename, taken from the URL."""
        return self.weights_url.rsplit("/", 1)[-1] or f"{self.name}-v{self.version}.tar.gz"

    @property
    def extract_dirname(self) -> str:
        """The subdirectory name inside the cache after extraction."""
        return f"{self.name}-v{self.version}"


@dataclass(frozen=True)
class ResolvedModel:
    """Local, on-disk paths for a fully-resolved (downloaded + extracted) model."""
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
    name = str(data["name"])
    if name != path.stem:
        raise RegistryError(
            f"model manifest {path}: name field {name!r} does not match "
            f"filename stem {path.stem!r}"
        )
    return ModelManifest(
        name=name,
        version=str(data["version"]),
        weights_url=str(data["weights_url"]),
        weights_sha256=str(data["weights_sha256"]),
        manifest_path=path,
        data=data,
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
# Placeholder detection
# --------------------------------------------------------------------------

def _looks_like_placeholder(sha256: str) -> bool:
    s = (sha256 or "").strip().lower()
    if not s or s in {"todo", "changeme", "unknown"}:
        return True
    if set(s) == {"0"} or set(s) == {"x"}:
        return True
    return False


# --------------------------------------------------------------------------
# Download / extract
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


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
    don't produce the expected top-level directory ``<name>-v<version>/``.
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


def _expected_files(extract_dir: Path) -> tuple[Path, Path]:
    return extract_dir / "weights.h5", extract_dir / "arch.yaml"


def resolve_model(name: str, *, force: bool = False) -> ResolvedModel:
    """Return a :class:`ResolvedModel` for ``name``, downloading if needed."""
    mf = load_manifest(name)

    if _looks_like_placeholder(mf.weights_sha256):
        raise RegistryError(
            f"model {name!r} has no verified checksum yet "
            f"(weights_sha256={mf.weights_sha256!r}). "
            "The maintainer needs to publish the release."
        )

    extract_dir = cache_dir() / mf.extract_dirname
    weights_path, arch_path = _expected_files(extract_dir)

    already_ok = (
        not force
        and weights_path.exists()
        and arch_path.exists()
    )
    if already_ok:
        return ResolvedModel(
            name=mf.name,
            version=mf.version,
            weights_path=weights_path,
            arch_config_path=arch_path,
            manifest=mf,
        )

    # (Re-)fetch: download the archive to a sibling temp filename, sha256
    # verify, then extract in place. If anything fails part-way, clean up
    # partial state so the next invocation retries cleanly.
    archive_path = cache_dir() / mf.archive_filename
    if force or not archive_path.exists() or _sha256_file(archive_path) != mf.weights_sha256:
        if archive_path.exists():
            archive_path.unlink()
        print(f"Downloading {mf.name} v{mf.version} from {mf.weights_url}",
              file=sys.stderr)
        _download(mf.weights_url, archive_path)

    got = _sha256_file(archive_path)
    if got != mf.weights_sha256:
        archive_path.unlink()
        raise RegistryError(
            f"SHA256 mismatch for {name!r} archive: expected "
            f"{mf.weights_sha256}, got {got}. File deleted."
        )

    # Remove any half-extracted state, then extract.
    if extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)
    _safe_extract(archive_path, cache_dir(), mf.extract_dirname)

    # Sanity check the extracted layout.
    if not weights_path.exists() or not arch_path.exists():
        raise RegistryError(
            f"model {name!r} archive did not contain the expected "
            f"weights.h5 and arch.yaml under {extract_dir}"
        )

    # Keep the archive around so `--force` can detect a matching download
    # without re-hitting the URL. Users can `drusilla models rm NAME` to
    # reclaim disk.
    return ResolvedModel(
        name=mf.name,
        version=mf.version,
        weights_path=weights_path,
        arch_config_path=arch_path,
        manifest=mf,
    )


def clear(name: str) -> list[Path]:
    """Delete cached artifacts for ``name``. Returns removed paths."""
    import shutil
    mf = load_manifest(name)
    removed: list[Path] = []
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
    """Describe the local cache state of ``name``."""
    mf = load_manifest(name)
    extract_dir = cache_dir() / mf.extract_dirname
    weights_path, arch_path = _expected_files(extract_dir)
    cached = weights_path.exists() and arch_path.exists()
    return {
        "name": name,
        "version": mf.version,
        "cached": cached,
        "extract_dir": str(extract_dir) if cached else None,
        "weights_path": str(weights_path) if cached else None,
        "arch_config_path": str(arch_path) if cached else None,
    }
