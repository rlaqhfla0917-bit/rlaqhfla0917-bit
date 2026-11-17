"""Leakage-safe preprocessing, classifiers, and metrics."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def create_models(k: int = 40) -> dict[str, Pipeline]:
    common = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=False)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=f_classif, k=k)),
    ]
    return {
        "Logistic Regression": Pipeline(common + [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline(common + [
            ("classifier", RandomForestClassifier(
                n_estimators=140, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED,
            )),
        ]),
    }


def screening_metrics(y_true, scores, fraction: float = 0.10) -> dict:
    """Metrics for reviewing a fixed proportion of the highest-scored samples."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or len(y) == 0 or not np.isfinite(s).all():
        raise ValueError("Invalid labels/scores")
    if not 0 < fraction <= 1:
        raise ValueError("Screening fraction must be in (0, 1]")
    n = len(y)
    budget = max(1, math.ceil(n * fraction))
    selected = np.argsort(-s, kind="stable")[:budget]
    selected_failures = int(y[selected].sum())
    total_failures = int(y.sum())
    prevalence = float(y.mean())
    precision_at_budget = selected_failures / budget
    recall_at_budget = selected_failures / total_failures if total_failures else None
    lift = precision_at_budget / prevalence if prevalence else None
    # AP is more interpretable for rare failures than raw accuracy.
    ap = float(average_precision_score(y, s)) if total_failures and total_failures != n else None
    roc = float(roc_auc_score(y, s)) if 0 < total_failures < n else None
    return {
        "samples": int(n), "failures": total_failures,
        "prevalence": prevalence, "review_fraction": fraction,
        "review_count": int(budget), "caught_failures": selected_failures,
        "precision_at_budget": float(precision_at_budget),
        "recall_at_budget": recall_at_budget,
        "lift_at_budget": lift, "average_precision": ap, "roc_auc": roc,
    }


def get_selected_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Interpret model associations, not causal fault origin."""
    imputer = model.named_steps["imputer"]
    if hasattr(imputer, "get_feature_names_out"):
        names = np.asarray(imputer.get_feature_names_out(feature_names))
    else:
        names = np.asarray(feature_names)[~np.isnan(imputer.statistics_)]
    names = names[model.named_steps["variance"].get_support()]
    names = names[model.named_steps["select"].get_support()]
    cls = model.named_steps["classifier"]
    if hasattr(cls, "feature_importances_"):
        values = cls.feature_importances_
        method = "random_forest_importance"
    else:
        values = np.abs(cls.coef_[0])
        method = "absolute_standardized_coefficient"
    return pd.DataFrame({"variable": names, "importance": values, "method": method}).sort_values("importance", ascending=False).reset_index(drop=True)
