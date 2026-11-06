"""GPU and system memory tracking utilities."""

from __future__ import annotations

import dataclasses
from typing import Any

import psutil

from indus.logging_utils import get_logger

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = get_logger(__name__)


@dataclasses.dataclass
class MemorySnapshot:
    """Point-in-time memory measurement."""

    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_peak_allocated_gb: float
    system_used_gb: float
    system_total_gb: float

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def _cuda_available() -> bool:
    """Check if CUDA is available. Returns False when torch is not installed."""
    return torch is not None and torch.cuda.is_available()


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current GPU and system memory usage.

    Returns:
        MemorySnapshot with current readings.
    """
    gpu_allocated = 0.0
    gpu_reserved = 0.0
    gpu_peak = 0.0

    if _cuda_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

    mem = psutil.virtual_memory()
    return MemorySnapshot(
        gpu_allocated_gb=round(gpu_allocated, 3),
        gpu_reserved_gb=round(gpu_reserved, 3),
        gpu_peak_allocated_gb=round(gpu_peak, 3),
        system_used_gb=round(mem.used / (1024**3), 3),
        system_total_gb=round(mem.total / (1024**3), 3),
    )


def log_memory(label: str = "current") -> MemorySnapshot:
    """Log and return a memory snapshot.

    Args:
        label: Descriptive label for the log message.

    Returns:
        MemorySnapshot captured at call time.
    """
    snap = get_memory_snapshot()
    logger.info(
        "Memory [%s] -- GPU allocated: %.2f GB, GPU peak: %.2f GB, System: %.2f / %.2f GB",
        label,
        snap.gpu_allocated_gb,
        snap.gpu_peak_allocated_gb,
        snap.system_used_gb,
        snap.system_total_gb,
    )
    return snap


def reset_peak_memory() -> None:
    """Reset the CUDA peak memory counter."""
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()
        logger.debug("CUDA peak memory stats reset")


def build_memory_report(
    before: MemorySnapshot,
    after: MemorySnapshot,
) -> dict[str, Any]:
    """Build a comparison report between two memory snapshots.

    Args:
        before: Snapshot taken before the operation.
        after: Snapshot taken after the operation.

    Returns:
        Dictionary with before, after, and delta values.
    """
    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "delta_gpu_allocated_gb": round(after.gpu_allocated_gb - before.gpu_allocated_gb, 3),
        "peak_gpu_gb": after.gpu_peak_allocated_gb,
    }
