"""Registry loading and cache-dir tests. Do NOT download anything."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from drusilla import registry


def test_bundled_registry_loads():
    entries = registry.load_registry()
    assert "vertebrates" in entries
    ve = entries["vertebrates"]
    assert ve.name == "vertebrates"
    assert ve.version
    assert ve.weights_url
    assert ve.config


def test_get_entry_unknown_model_raises():
    with pytest.raises(registry.RegistryError):
        registry.get_entry("this-model-does-not-exist")


def test_cache_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    d = registry.cache_dir()
    assert d == tmp_path / "models"
    assert d.exists()


def test_placeholder_checksum_rejected(tmp_path, monkeypatch):
    """A model whose checksum is still 'TODO' must refuse to resolve."""
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    ve = registry.get_entry("vertebrates")
    if not registry._looks_like_placeholder(ve.weights_sha256):
        pytest.skip("vertebrates release has real checksum; test not applicable")
    with pytest.raises(registry.RegistryError):
        registry.resolve_weights("vertebrates")


def test_config_resolves():
    p = registry.resolve_config("vertebrates")
    assert p.exists()
    assert p.suffix == ".yaml"


def test_status_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUSILLA_CACHE_DIR", str(tmp_path))
    st = registry.local_status("vertebrates")
    assert st["cached"] is False
