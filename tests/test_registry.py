"""Registry loading and cache-dir tests. Do NOT download anything."""

from __future__ import annotations

import hashlib
import os
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
    assert mf.weights_sha256


def test_load_manifest_unknown_raises():
    with pytest.raises(registry.RegistryError):
        registry.load_manifest("this-model-does-not-exist")


def test_load_manifest_from_user_dir(tmp_path: Path, monkeypatch):
    """DRUSILLA_MODEL_CFG_DIR should register additional manifests."""
    (tmp_path / "custom.yaml").write_text(
        "name: custom\n"
        "version: '0.1'\n"
        "weights_url: 'https://example.com/custom-v0.1.tar.gz'\n"
        "weights_sha256: 'deadbeef'\n",
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
        "weights_url: 'x'\n"
        "weights_sha256: 'x'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(tmp_path))
    with pytest.raises(registry.RegistryError):
        registry.load_manifest("wrongname")


def test_manifest_missing_required_field_rejected(tmp_path: Path, monkeypatch):
    (tmp_path / "broken.yaml").write_text(
        "name: broken\nversion: '1'\nweights_url: 'x'\n",   # no sha256
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


# ---------- placeholder rejection ----------

def test_placeholder_checksum_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    mf = registry.load_manifest("vertebrates")
    if not registry._looks_like_placeholder(mf.weights_sha256):
        pytest.skip("vertebrates release has real checksum; test not applicable")
    with pytest.raises(registry.RegistryError):
        registry.resolve_model("vertebrates")


# ---------- status ----------

def test_status_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    st = registry.local_status("vertebrates")
    assert st["cached"] is False


# ---------- resolve_model end-to-end with a fake local archive ----------

def _make_fake_archive(dir_name: str, workdir: Path) -> tuple[Path, str]:
    """Build ``<dir_name>.tar.gz`` containing ``<dir_name>/weights.h5`` and
    ``<dir_name>/arch.yaml``. Returns (archive_path, sha256hex)."""
    extract_root = workdir / dir_name
    extract_root.mkdir()
    (extract_root / "weights.h5").write_bytes(b"fake weights payload")
    (extract_root / "arch.yaml").write_text(
        "data: {chunk_len: 9999}\nmodel: {type: cnn_lstm}\n"
    )
    archive = workdir / f"{dir_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(extract_root, arcname=dir_name)
    h = hashlib.sha256()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return archive, h.hexdigest()


def _write_manifest(dir_path: Path, name: str, version: str, url: str, sha: str):
    (dir_path / f"{name}.yaml").write_text(
        f"name: {name}\n"
        f"version: '{version}'\n"
        f"weights_url: '{url}'\n"
        f"weights_sha256: '{sha}'\n",
        encoding="utf-8",
    )


def test_resolve_model_downloads_and_extracts(tmp_path: Path, monkeypatch):
    """End-to-end: manifest + tar.gz -> extracted cache with correct paths."""
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive, sha = _make_fake_archive("fakemodel-v1.0", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(
        cfg_dir, name="fakemodel", version="1.0",
        url=archive.as_uri(),   # file:// URL so urlopen works locally
        sha=sha,
    )
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
    archive, sha = _make_fake_archive("fakemodel-v1.0", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri(), sha)
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    r1 = registry.resolve_model("fakemodel")
    mtime = r1.weights_path.stat().st_mtime
    r2 = registry.resolve_model("fakemodel")
    assert r2.weights_path.stat().st_mtime == mtime


def test_resolve_model_bad_sha_rejected(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive, _real_sha = _make_fake_archive("fakemodel-v1.0", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(
        cfg_dir, "fakemodel", "1.0", archive.as_uri(),
        "0" * 64,   # non-placeholder but wrong
    )
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    # sha of all-zeros is detected as placeholder; use a real-looking mismatch
    _write_manifest(
        cfg_dir, "fakemodel", "1.0", archive.as_uri(),
        "a" * 64,
    )
    with pytest.raises(registry.RegistryError):
        registry.resolve_model("fakemodel")


def test_clear_removes_extract_dir(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "release"
    workdir.mkdir()
    archive, sha = _make_fake_archive("fakemodel-v1.0", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri(), sha)
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
    archive, sha = _make_fake_archive("fakemodel-v1.0", workdir)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri(), sha)
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
    """An archive with a member that escapes the extract dir must be rejected."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    # Build an evil tar with a member "../evil"
    workdir = tmp_path / "release"
    workdir.mkdir()
    good = workdir / "fakemodel-v1.0"
    good.mkdir()
    (good / "weights.h5").write_bytes(b"ok")
    (good / "arch.yaml").write_text("data: {}\nmodel: {}\n")
    (workdir / "evil").write_text("pwned")

    archive = workdir / "fakemodel-v1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(good, arcname="fakemodel-v1.0")
        tf.add(workdir / "evil", arcname="../evil")

    h = hashlib.sha256()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)

    _write_manifest(cfg_dir, "fakemodel", "1.0", archive.as_uri(), h.hexdigest())
    monkeypatch.setenv("DRUSILLA_MODEL_CFG_DIR", str(cfg_dir))
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(registry.RegistryError):
        registry.resolve_model("fakemodel")
