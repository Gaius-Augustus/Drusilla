"""Registry loading and cache-dir tests. Do NOT download anything."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from drusilla import registry


# ---------- bundled manifest discovery ----------

def test_bundled_manifest_loads():
    manifests = registry.list_manifests()
    assert "vertebrates" in manifests
    mf = manifests["vertebrates"]
    assert mf.name == "vertebrates"
    assert mf.version
    assert mf.weights_url


def test_load_manifest_unknown_raises():
    with pytest.raises(registry.RegistryError):
        registry.load_manifest("this-model-does-not-exist")


def test_load_manifest_from_user_dir(tmp_path: Path, monkeypatch):
    """DRUSILLA_MODEL_CFG_DIR should register additional manifests."""
    (tmp_path / "custom.yaml").write_text(
        "name: custom\n"
        "version: '0.1'\n"
        "weights_url: 'https://example.com/custom-v0.1.tar.gz'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(tmp_path))
    manifests = registry.list_manifests()
    assert "custom" in manifests
    assert manifests["custom"].version == "0.1"


def test_manifest_filename_stem_mismatch_rejected(tmp_path: Path, monkeypatch):
    (tmp_path / "wrongname.yaml").write_text(
        "name: differentname\n"
        "version: '1'\n"
        "weights_url: 'x'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(tmp_path))
    with pytest.raises(registry.RegistryError):
        registry.load_manifest("wrongname")


def test_manifest_missing_required_field_rejected(tmp_path: Path, monkeypatch):
    (tmp_path / "broken.yaml").write_text(
        "name: broken\nversion: '1'\n",   # no weights_url
        encoding="utf-8",
    )
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(tmp_path))
    with pytest.raises(registry.RegistryError):
        registry.load_manifest("broken")


# ---------- cache dir ----------

def test_cache_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    d = registry.cache_dir()
    assert d == tmp_path / "models"
    assert d.exists()


# ---------- status ----------

def test_status_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    st = registry.local_status("vertebrates")
    assert st["cached"] is False


# ---------- resolve_model end-to-end with a fake local archive ----------

def _make_fake_archive(dir_name: str, workdir: Path) -> Path:
    """Build ``<dir_name>.tar.gz`` containing ``<dir_name>/weights.h5`` and
    ``<dir_name>/arch.yaml``. Returns the archive path."""
    extract_root = workdir / dir_name
    extract_root.mkdir()
    (extract_root / "weights.h5").write_bytes(b"fake weights payload")
    (extract_root / "arch.yaml").write_text(
        "data: {chunk_len: 9999}\nmodel: {type: cnn_lstm}\n"
    )
    archive = workdir / f"{dir_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(extract_root, arcname=dir_name)
    return archive


def _write_manifest(dir_path: Path, name: str, version: str, url: str):
    (dir_path / f"{name}.yaml").write_text(
        f"name: {name}\n"
        f"version: '{version}'\n"
        f"weights_url: '{url}'\n",
        encoding="utf-8",
    )


def test_resolve_model_downloads_and_extracts(tmp_path: Path, monkeypatch):
    """End-to-end: manifest + tar.gz -> extracted cache with correct paths."""
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    resolved = registry.resolve_model("fakemodel")
    assert resolved.name == "fakemodel"
    assert resolved.version == "1.0"
    assert resolved.weights_path.exists()
    assert resolved.arch_config_path.exists()
    assert resolved.weights_path.read_bytes() == b"fake weights payload"
    assert "chunk_len" in resolved.arch_config_path.read_text()


def test_resolve_model_second_call_uses_cache(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    r1 = registry.resolve_model("fakemodel")
    mtime = r1.weights_path.stat().st_mtime
    r2 = registry.resolve_model("fakemodel")
    assert r2.weights_path.stat().st_mtime == mtime


def test_resolve_model_force_redownloads(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    # Instrument the downloader to count invocations.
    calls: list[str] = []
    real_download = registry._download

    def counting_download(url, dest):
        calls.append(url)
        return real_download(url, dest)

    monkeypatch.setattr(registry, "_download", counting_download)

    registry.resolve_model("fakemodel")
    registry.resolve_model("fakemodel")                # cache hit, no download
    registry.resolve_model("fakemodel", force=True)    # forced re-download
    assert len(calls) == 2


def test_resolve_model_reextracts_on_version_bump(tmp_path: Path, monkeypatch):
    """Bumping the manifest version should trigger a re-download."""
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    calls: list[str] = []
    real_download = registry._download

    def counting_download(url, dest):
        calls.append(url)
        return real_download(url, dest)

    monkeypatch.setattr(registry, "_download", counting_download)

    registry.resolve_model("fakemodel")
    registry.resolve_model("fakemodel")   # cached: no download
    assert len(calls) == 1

    # Bump version in the manifest.
    _write_manifest(cfg_dir, "fakemodel", "1.1", archive.as_uri())
    registry.resolve_model("fakemodel")   # version mismatch: re-download
    assert len(calls) == 2


def test_resolve_model_corrupted_archive_cleans_up(tmp_path: Path, monkeypatch):
    """A junk archive should raise and get deleted so the next call retries."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a real gzip stream")
    _write_manifest(cfg_dir, "fakemodel", "1.0", junk.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(Exception):
        registry.resolve_model("fakemodel")
    # After the failure the copy in the cache should be gone so the
    # next call will re-fetch instead of looping on a broken file.
    cache_copy = (tmp_path / "cache" / "models" / "junk.tar.gz")
    assert not cache_copy.exists()


def test_clear_removes_extract_dir(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    r = registry.resolve_model("fakemodel")
    assert r.weights_path.exists()
    removed = registry.clear("fakemodel")
    assert len(removed) >= 1
    assert not r.weights_path.exists()


def test_local_status_after_resolve(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive = _make_fake_archive("fakemodel", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    st = registry.local_status("fakemodel")
    assert st["cached"] is False
    registry.resolve_model("fakemodel")
    st = registry.local_status("fakemodel")
    assert st["cached"] is True
    assert st["weights_path"] is not None
    assert st["arch_config_path"] is not None


def test_tar_slip_rejected(tmp_path: Path, monkeypatch):
    """An archive whose members escape the extract dir must be rejected."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    workdir = tmp_path / "release"
    workdir.mkdir()
    good = workdir / "fakemodel"
    good.mkdir()
    (good / "weights.h5").write_bytes(b"ok")
    (good / "arch.yaml").write_text("data: {}\nmodel: {}\n")
    (workdir / "evil").write_text("pwned")

    archive = workdir / "fakemodel.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(good, arcname="fakemodel")
        tf.add(workdir / "evil", arcname="../evil")

    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(Exception):
        registry.resolve_model("fakemodel")
