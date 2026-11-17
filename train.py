"""Train on chronological SECOM partitions and evaluate a never-tuned holdout."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data import load_secom, chronological_indices
from src.modeling import create_models, screening_metrics, get_selected_feature_importance


def run(raw_dir: str | Path, out_dir: str | Path, k: int = 40) -> dict:
    x, y, meta = load_secom(raw_dir)
    if x.shape[1] < k:
        raise ValueError(f"Only {x.shape[1]} features, k={k}")
    train, validation, test = chronological_indices(len(x))
    for phase, idx in [("training", train), ("validation", validation), ("test", test)]:
        failures = int(y.iloc[idx].sum())
        if failures < 1 or failures == len(y.iloc[idx]):
            raise ValueError(f"No binary class variation in {phase} period")
    models = create_models(k=k)
    comparison = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
        for name, model in models.items():
            model.fit(x.iloc[train], y.iloc[train])
            predicted = model.predict_proba(x.iloc[validation])[:, 1]
            m = screening_metrics(y.iloc[validation], predicted)
            comparison.append({"model": name, **m})
            print(f"Validation {name}: Recall@10%={m['recall_at_budget']:.3f}, AP={m['average_precision']:.3f}")
    # Rank by actual discovered failures at fixed review budget, then AP. Validation only.
    comparison.sort(key=lambda m: (m["recall_at_budget"], m["average_precision"]), reverse=True)
    winning_name = comparison[0]["model"]
    fitted = clone(models[winning_name])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
        fitted.fit(x.iloc[:validation.stop], y.iloc[:validation.stop])
    scores = fitted.predict_proba(x.iloc[test])[:, 1]
    metrics = screening_metrics(y.iloc[test], scores)
    if not 0 <= metrics["average_precision"] <= 1:
        raise ValueError("Invalid AP")
    importance = get_selected_feature_importance(fitted, list(x.columns))
    predictions = meta.iloc[test].copy()
    predictions["true_fail"] = y.iloc[test].to_numpy()
    predictions["risk_score"] = scores
    predictions = predictions.sort_values("risk_score", ascending=False).reset_index(drop=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, out / "model.joblib")
    predictions.to_csv(out / "test_predictions.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    doc = {
        "model_name": winning_name,
        "source": "UCI SECOM; McCann and Johnston (2008); CC BY 4.0",
        "dataset_doi": "10.24432/C54305",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split": "chronological 60% train / 20% validation / 20% test; fit selection on train+validation",
        "selection": "Highest validation Recall@Top10%, then Average Precision; test untouched until final evaluation",
        "features": int(x.shape[1]),
        "total_samples": int(len(x)),
        "total_failures": int(y.sum()),
        "null_cells": int(x.isna().sum().sum()),
        "missing_fraction": float(x.isna().to_numpy().mean()),
        "train_samples": int(train.stop),
        "validation_samples": int(validation.stop - validation.start),
        "test_samples": int(test.stop - test.start),
        "validation_results": comparison,
        "test_metrics": metrics,
        "test_period": {"start": str(meta.iloc[test]["timestamp"].min()), "end": str(meta.iloc[test]["timestamp"].max())},
        "disclaimer": "Retrospective screening prototype. Anonymized features, no verified causal root cause, no live fab connection; outputs are uncalibrated relative rankings, not defect probabilities.",
    }
    (out / "metrics.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Held-out TEST | model={winning_name} | Recall@10%={metrics['recall_at_budget']:.3f} | Lift@10%={metrics['lift_at_budget']:.2f} | AP={metrics['average_precision']:.3f}")
    print(f"Saved model, metrics, predictions and feature importance to {out}")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--features", default=40, type=int)
    args = parser.parse_args()
    try:
        run(args.raw_dir, args.out_dir, args.features)
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
