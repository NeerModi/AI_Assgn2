from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_global_seeds(seed: int) -> None:
    """Sets numpy seed for reproducibility."""
    np.random.seed(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Loads YAML config as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a top-level mapping")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Creates directory if missing and returns Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with same-length output."""
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
