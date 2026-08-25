"""Model registry: locate, download, and verify Drusilla model weights.

The registry itself is a small YAML document that ships with the package
at ``drusilla/_bundled/models.yaml``. It maps a short model name (e.g.
``vertebrates``) to a download URL, SHA256 checksum, and the training
config filename to use for architecture reconstruction.

Runtime overrides:

* ``DRUSILLA_MODELS_URL``  - fetch the registry YAML from this URL instead
  of the bundled copy. Useful for testing new releases without repackaging.
* ``DRUSILLA_CACHE_DIR``   - override the cache directory
  (default: ``$XDG_CACHE_HOME/drusilla`` or ``~/.cache/drusilla``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_RESOURCE = "models.yaml"
_CHUNK = 1 << 20  # 1 MiB read chunks for streaming download


@dataclass(frozen=True)
class ModelEntry:
    name: str
    version: str
    weights_url: str
    weights_sha256: str
    config: str
    clades: list[str]
    description: str

    @property
    def filename(self) -> str:
        """Canonical local filename for the cached weights."""
        return f"{self.name}-v{self.version}.weights.h5"


class RegistryError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Cache location
# --------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the Drusilla cache directory (created on demand)."""
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
# Registry loading
# --------------------------------------------------------------------------

def _repo_root_candidates() -> list[Path]:
    """Editable-install fallback: look for repo-root models.yaml / configs."""
    here = Path(__file__).resolve()
    return [here.parents[2], here.parents[3]]


def _bundled_registry_path() -> Path:
    p = Path(__file__).parent / "_bundled" / DEFAULT_REGISTRY_RESOURCE
    if p.exists():
        return p
    for root in _repo_root_candidates():
        alt = root / DEFAULT_REGISTRY_RESOURCE
        if alt.exists():
            return alt
    return p


def _load_registry_yaml() -> dict[str, Any]:
    url = os.environ.get("DRUSILLA_MODELS_URL")
    if url:
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                text = fh.read().decode("utf-8")
        except Exception as e:
            raise RegistryError(f"failed to fetch DRUSILLA_MODELS_URL={url}: {e}") from e
        return yaml.safe_load(text) or {}
    path = _bundled_registry_path()
    if not path.exists():
        raise RegistryError(
            f"bundled registry not found at {path}. "
            "Reinstall the package or set DRUSILLA_MODELS_URL."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_registry() -> dict[str, ModelEntry]:
    """Parse the registry YAML into ``{name: ModelEntry}``."""
    doc = _load_registry_yaml()
    entries: dict[str, ModelEntry] = {}
    for name, meta in (doc.get("models") or {}).items():
        try:
            entries[name] = ModelEntry(
                name=name,
                version=str(meta["version"]),
                weights_url=str(meta["weights_url"]),
                weights_sha256=str(meta["weights_sha256"]),
                config=str(meta.get("config", "default.yaml")),
                clades=list(meta.get("clades", []) or []),
                description=str(meta.get("description", "")),
            )
        except KeyError as e:
            raise RegistryError(
                f"model {name!r} missing required field {e}"
            ) from e
    return entries


def get_entry(name: str) -> ModelEntry:
    entries = load_registry()
    if name not in entries:
        available = ", ".join(sorted(entries)) or "(none)"
        raise RegistryError(
            f"unknown model {name!r}. Available: {available}"
        )
    return entries[name]


# --------------------------------------------------------------------------
# Download / verify
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_placeholder(sha256: str) -> bool:
    """Detect obvious placeholder hashes so we fail with a clear message."""
    s = sha256.strip().lower()
    if not s or s in {"todo", "changeme", "unknown"}:
        return True
    if set(s) == {"0"} or set(s) == {"x"}:
        return True
    return False


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest`` (writes to ``dest.part`` and renames)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = resp.length
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
        print("", file=sys.stderr)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def resolve_weights(name: str, *, force: bool = False) -> Path:
    """Return the local path to ``name``'s weights, downloading if needed.

    If the file already exists and its SHA256 matches the registry entry,
    it is reused. If ``force`` is True, the cached copy is re-downloaded.
    """
    entry = get_entry(name)
    if _looks_like_placeholder(entry.weights_sha256):
        raise RegistryError(
            f"model {name!r} has no verified checksum yet "
            f"(weights_sha256={entry.weights_sha256!r}). "
            "The maintainer needs to publish the release."
        )
    dest = cache_dir() / entry.filename
    if dest.exists() and not force:
        if _sha256_file(dest) == entry.weights_sha256:
            return dest
        print(
            f"warning: cached {dest} does not match expected checksum; "
            "redownloading.", file=sys.stderr,
        )
        dest.unlink()
    print(f"Downloading {entry.name} v{entry.version} from {entry.weights_url}",
          file=sys.stderr)
    _download(entry.weights_url, dest)
    got = _sha256_file(dest)
    if got != entry.weights_sha256:
        dest.unlink()
        raise RegistryError(
            f"SHA256 mismatch for {name!r}: expected "
            f"{entry.weights_sha256}, got {got}. File deleted."
        )
    return dest


def resolve_config(name: str) -> Path:
    """Return the path to the bundled training config for ``name``."""
    entry = get_entry(name)
    candidates = [Path(__file__).parent / "_bundled" / "configs" / entry.config]
    for root in _repo_root_candidates():
        candidates.append(root / "configs" / entry.config)
    for p in candidates:
        if p.exists():
            return p
    raise RegistryError(
        f"config {entry.config!r} for model {name!r} not found; "
        f"looked in {[str(p) for p in candidates]}"
    )


def clear(name: str) -> Path | None:
    """Delete the cached weights file for ``name``. Returns the removed path or None."""
    entry = get_entry(name)
    dest = cache_dir() / entry.filename
    if dest.exists():
        dest.unlink()
        return dest
    return None


def local_status(name: str) -> dict[str, Any]:
    """Return a small dict describing the local cache state of ``name``."""
    entry = get_entry(name)
    dest = cache_dir() / entry.filename
    if not dest.exists():
        return {"name": name, "version": entry.version, "cached": False, "path": None}
    sha = _sha256_file(dest)
    ok = (sha == entry.weights_sha256) if not _looks_like_placeholder(entry.weights_sha256) else None
    return {
        "name": name,
        "version": entry.version,
        "cached": True,
        "path": str(dest),
        "sha256_ok": ok,
    }
