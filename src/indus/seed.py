"""Reproducibility utilities for deterministic training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np

from indus.logging_utils import get_logger

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d across all libraries", seed)
