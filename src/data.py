"""UCI SECOM data loading with explicit time ordering and labels."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RAW_FILES = ("secom.data", "secom_labels.data")


def load_secom(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {raw_dir}. "
            "Run 'python scripts/download_data.py' (internet required)."
        )
    x = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(
        raw_dir / "secom_labels.data", sep=r"\s+", header=None,
        names=["label", "date", "time"], engine="python",
    )
    # The raw label file contains label, date, and time tokens, e.g. -1 19/07/2008 11:55:00.
    if labels.shape[1] != 3:
        raise ValueError("Unexpected label format: expected label, date, and time tokens")
    if len(x) != len(labels):
        raise ValueError(f"Row mismatch: {len(x)} feature rows and {len(labels)} labels")
    labels["label"] = pd.to_numeric(labels["label"], errors="raise")
    if not set(labels["label"].dropna().unique()).issubset({-1, 1}):
        raise ValueError("UCI labels must be -1 (pass) or 1 (fail)")
    timestamps = pd.to_datetime(
        labels["date"].astype(str) + " " + labels["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S", errors="coerce",
    )
    if timestamps.isna().any():
        # Some mirrors use month-first formatting; fail explicitly rather than silently reorder dates.
        raise ValueError("Failed to parse UCI timestamps as DD/MM/YYYY HH:MM:SS")
    x.columns = [f"sensor_{i:03d}" for i in range(1, x.shape[1] + 1)]
    x = x.apply(pd.to_numeric, errors="coerce")
    y = labels["label"].eq(1).astype(int)
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    metadata = pd.DataFrame({
        "sample_id": [f"SECOM-{i + 1:04d}" for i in order],
        "timestamp": timestamps.iloc[order].to_numpy(),
    })
    return x.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), metadata


def chronological_indices(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[slice, slice, slice]:
    if n < 30 or not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Invalid split configuration")
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return slice(0, a), slice(a, b), slice(b, n)
