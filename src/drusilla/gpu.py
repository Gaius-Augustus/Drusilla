"""GPU introspection helpers for automatic batch-size selection.

Mirrors the Tiberius approach (``tiberius.main.get_gpu_memory_gb`` /
``compute_auto_batch_size``) but recalibrated for Drusilla's much shorter
default ``chunk_len`` of ~10k (vs Tiberius's ~500k) and for the CNN-BiLSTM
architecture used by the vertebrate / embryophyta releases.

Empirical anchor: chunk_len=9999 with the default cnn_lstm architecture
fits batch_size=200 on a 40 GB A100, ~400 on an 80 GB A100. All other
anchors are conservative extrapolations from that point.
"""

from __future__ import annotations

import logging
import subprocess as sp

log = logging.getLogger(__name__)


_DEFAULT_FALLBACK_BATCH_SIZE = 8


def get_gpu_memory_gb(device_index: int = 0) -> float | None:
    """Return total memory of the selected NVIDIA GPU in GB, or ``None``
    if ``nvidia-smi`` is unavailable / the device index is out of range.

    On failure, callers should fall back to a hard-coded default rather
    than raising.
    """
    try:
        result = sp.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log.warning(
            "nvidia-smi not found. Cannot auto-detect GPU memory. "
            "Batch size will fall back to %d.",
            _DEFAULT_FALLBACK_BATCH_SIZE,
        )
        return None
    except sp.CalledProcessError as exc:
        log.warning(
            "Failed to query GPU memory with nvidia-smi: %s. "
            "Batch size will fall back to %d.",
            exc, _DEFAULT_FALLBACK_BATCH_SIZE,
        )
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if device_index >= len(lines):
        log.warning(
            "GPU index %d out of range. Found %d GPU(s). "
            "Batch size will fall back to %d.",
            device_index, len(lines), _DEFAULT_FALLBACK_BATCH_SIZE,
        )
        return None

    memory_mb = float(lines[device_index])
    return memory_mb / 1024.0


def compute_auto_batch_size(
    chunk_len: int,
    gpu_memory_gb: float,
    safety_factor: float = 0.9,
    min_batch_size: int = 1,
    max_batch_size: int | None = None,
) -> int:
    """Estimate an inference batch size from GPU memory and chunk length.

    The estimate is linear in ``gpu_memory_gb / chunk_len`` around a
    reference anchor (a workload known to fit) and then snapped to a
    preferred value nearby for numerical friendliness.

    Reference anchors (empirical):

    * >= 70 GB    ->  chunk_len=9999,  batch=400   (~80 GB A100)
    * >= 30 GB    ->  chunk_len=9999,  batch=200   (~40 GB A100)
    * >= 12 GB    ->  chunk_len=9999,  batch=64
    * <  12 GB    ->  chunk_len=9999,  batch=16
    """
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be > 0, got {chunk_len}")
    if gpu_memory_gb <= 0:
        raise ValueError(f"gpu_memory_gb must be > 0, got {gpu_memory_gb}")
    if safety_factor <= 0:
        raise ValueError(f"safety_factor must be > 0, got {safety_factor}")

    if gpu_memory_gb >= 70:
        ref_mem, ref_bs = 80.0, 400
    elif gpu_memory_gb >= 30:
        ref_mem, ref_bs = 40.0, 200
    elif gpu_memory_gb >= 12:
        ref_mem, ref_bs = 16.0, 64
    else:
        ref_mem, ref_bs = 8.0, 16

    ref_chunk_len = 9999
    estimated = (
        (gpu_memory_gb / ref_mem)
        * ref_bs
        * (ref_chunk_len / chunk_len)
        * safety_factor
    )

    batch_size = max(min_batch_size, estimated)

    # Snap to a nearby preferred size if close, so we don't emit odd
    # numbers like 137. Preferred sizes cover typical GPU sweet spots.
    preferred_batch_sizes = [
        1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160,
        200, 256, 320, 400, 512, 640, 800, 1024,
    ]
    max_deviation = max(4, int(0.05 * batch_size))
    candidates = [b for b in preferred_batch_sizes if abs(b - batch_size) < max_deviation]
    if candidates:
        batch_size = min(candidates, key=lambda x: abs(x - batch_size))

    if max_batch_size is not None:
        batch_size = min(batch_size, max_batch_size)
    batch_size = int(batch_size + 0.5)
    if batch_size < min_batch_size:
        batch_size = min_batch_size
    return batch_size


def resolve_batch_size(chunk_len: int) -> tuple[int, str]:
    """High-level helper: return ``(batch_size, source_description)``.

    Tries GPU detection; falls back to a fixed default. The description
    string is suitable for logging to the user.
    """
    mem_gb = get_gpu_memory_gb()
    if mem_gb is None:
        return _DEFAULT_FALLBACK_BATCH_SIZE, (
            f"auto (fallback default; nvidia-smi unavailable or no GPU)"
        )
    bs = compute_auto_batch_size(chunk_len=chunk_len, gpu_memory_gb=mem_gb)
    return bs, f"auto (gpu_memory={mem_gb:.1f} GB, chunk_len={chunk_len})"
