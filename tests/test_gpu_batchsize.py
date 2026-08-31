"""Tests for automatic batch-size selection. No CUDA, no TF."""

from __future__ import annotations

import subprocess as sp
from unittest.mock import patch

import pytest

from drusilla.gpu import (
    compute_auto_batch_size,
    get_gpu_memory_gb,
    resolve_batch_size,
    _DEFAULT_FALLBACK_BATCH_SIZE,
)


# ---------- compute_auto_batch_size ----------

def test_anchor_40gb_9999_returns_200():
    # Empirical anchor: 40 GB A100 at chunk_len=9999 -> batch_size=200.
    bs = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=40.0,
                                 safety_factor=1.0)
    assert bs == 200


def test_anchor_80gb_9999_returns_400():
    bs = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=80.0,
                                 safety_factor=1.0)
    assert bs == 400


def test_scales_down_with_chunk_len():
    # Doubling chunk_len should roughly halve the batch size.
    bs_short = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=40.0,
                                       safety_factor=1.0)
    bs_long  = compute_auto_batch_size(chunk_len=20000, gpu_memory_gb=40.0,
                                       safety_factor=1.0)
    assert bs_long < bs_short
    # Within a factor of 2 tolerance for the snap-to-preferred behaviour.
    assert bs_long <= bs_short


def test_scales_up_with_gpu_mem():
    bs_small = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=16.0)
    bs_big   = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=80.0)
    assert bs_big > bs_small


def test_min_batch_size_enforced():
    # Tiny GPU + huge chunk_len should still return >= 1.
    bs = compute_auto_batch_size(chunk_len=1_000_000, gpu_memory_gb=6.0)
    assert bs >= 1


def test_max_batch_size_enforced():
    bs = compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=80.0,
                                 max_batch_size=64)
    assert bs <= 64


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_auto_batch_size(chunk_len=0, gpu_memory_gb=40.0)
    with pytest.raises(ValueError):
        compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=0)
    with pytest.raises(ValueError):
        compute_auto_batch_size(chunk_len=9999, gpu_memory_gb=40.0,
                                safety_factor=0)


# ---------- get_gpu_memory_gb (subprocess mocked) ----------

class _MockCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_get_gpu_memory_gb_parses_nvidia_smi():
    with patch("subprocess.run",
               return_value=_MockCompletedProcess("40960\n")):
        mem = get_gpu_memory_gb()
    assert mem == pytest.approx(40960 / 1024.0)


def test_get_gpu_memory_gb_selects_by_device_index():
    with patch("subprocess.run",
               return_value=_MockCompletedProcess("16384\n81920\n")):
        assert get_gpu_memory_gb(device_index=0) == pytest.approx(16.0)
        assert get_gpu_memory_gb(device_index=1) == pytest.approx(80.0)


def test_get_gpu_memory_gb_out_of_range_returns_none():
    with patch("subprocess.run",
               return_value=_MockCompletedProcess("40960\n")):
        assert get_gpu_memory_gb(device_index=3) is None


def test_get_gpu_memory_gb_missing_nvidia_smi_returns_none():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        assert get_gpu_memory_gb() is None


def test_get_gpu_memory_gb_process_error_returns_none():
    err = sp.CalledProcessError(returncode=1, cmd=["nvidia-smi"])
    with patch("subprocess.run", side_effect=err):
        assert get_gpu_memory_gb() is None


# ---------- resolve_batch_size ----------

def test_resolve_batch_size_uses_gpu_mem():
    # resolve_batch_size applies safety_factor=0.9 by default, so the
    # 40 GB anchor (200 at safety=1.0) lands slightly below 200.
    with patch("drusilla.gpu.get_gpu_memory_gb", return_value=40.0):
        bs, desc = resolve_batch_size(chunk_len=9999)
    assert 150 <= bs <= 200
    assert "gpu_memory=40.0 GB" in desc


def test_resolve_batch_size_fallback():
    with patch("drusilla.gpu.get_gpu_memory_gb", return_value=None):
        bs, desc = resolve_batch_size(chunk_len=9999)
    assert bs == _DEFAULT_FALLBACK_BATCH_SIZE
    assert "fallback" in desc.lower()
