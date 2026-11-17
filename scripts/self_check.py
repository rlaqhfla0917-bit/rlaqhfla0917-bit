"""Offline integration test using SYNTHETIC, NOT real semiconductor, data."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import run
from src.modeling import screening_metrics


def main():
    rng = np.random.default_rng(42)
    n, p = 340, 65
    x = rng.normal(0, 1, size=(n, p))
    prob = 1 / (1 + np.exp(-(-2.5 + 1.5 * x[:, 4] - 1.2 * x[:, 8])))
    y = (rng.random(n) < prob).astype(int)
    x[:, 1] = 0.0  # constant
    x[::12, 2] = np.nan
    x[:, 45] = np.nan  # all missing
    stamp = pd.date_range("2008-07-01", periods=n, freq="h")
    with tempfile.TemporaryDirectory(prefix="fabpulse_test_") as t:
        root = Path(t)
        raw = root / "raw"
        raw.mkdir()
        np.savetxt(raw / "secom.data", x, fmt="%.8f")
        with (raw / "secom_labels.data").open("w") as f:
            for label, dt in zip(y, stamp):
                f.write(f"{1 if label else -1} {dt:%d/%m/%Y %H:%M:%S}\n")
        metrics = run(raw, root / "artifacts", k=20)
        assert metrics["test_metrics"]["samples"] == len(range(int(n * .8), n))
        assert (root / "artifacts" / "model.joblib").exists()
        assert (root / "artifacts" / "test_predictions.csv").exists()
        assert screening_metrics([0, 1, 0, 1], [.1, .9, .2, .8], .5)["caught_failures"] == 2
        print("PASS: end-to-end training and reports on SYNTHETIC test data")


if __name__ == "__main__":
    main()
