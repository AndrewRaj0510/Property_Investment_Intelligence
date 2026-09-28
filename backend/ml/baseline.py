"""Naive baseline predictors for the yield model comparison.

A model's R²/MAE means little in isolation - it must beat a trivial predictor an
analyst could use with no machine learning. Two naive baselines are evaluated
under the *identical* 5-fold CV protocol as the models (same folds, same metrics)
so the comparison is fair:

    * ``global_median`` - predict every bucket's yield as the median yield of the
      training fold. The absolute floor: knowing nothing about the property.
    * ``area_type_median`` - predict a bucket's yield as the median yield of its
      (area, property_type) group in the training fold, falling back to the
      global training median for groups unseen in that fold. This is what a
      sensible human analyst would do - "what do similar properties yield?" - and
      is the meaningful bar the ML model must clear.

The gap between ``area_type_median`` and the trained model quantifies the value
the model adds *beyond* a simple lookup table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ml.train import TARGET, _load_rows

log = structlog.get_logger(__name__)


def _cv_metrics(y_true: list, y_pred: list) -> dict:
    yt, yp = np.array(y_true), np.array(y_pred)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r_squared": float(r2_score(yt, yp)),
    }


def evaluate_baselines(
    feature_version: int | None = None, target: str = TARGET
) -> dict[str, dict]:
    """Return CV metrics for the naive baselines on the same folds as the models."""
    df = pd.DataFrame(_load_rows(feature_version))
    df = df[df[target].notna()].reset_index(drop=True)
    y = df[target].astype(float).values
    keys = list(zip(df["community"], df["property_type"]))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    gm_fold, at_fold = [], []
    for train_idx, test_idx in kf.split(df):
        train_y = y[train_idx]
        global_med = float(np.median(train_y))

        group_med: dict[tuple, float] = {}
        buckets: dict[tuple, list] = {}
        for i in train_idx:
            buckets.setdefault(keys[i], []).append(y[i])
        for k, vals in buckets.items():
            group_med[k] = float(np.median(vals))

        y_test = y[test_idx]
        gm_pred = [global_med] * len(test_idx)
        at_pred = [group_med.get(keys[i], global_med) for i in test_idx]

        gm_fold.append(_cv_metrics(y_test, gm_pred))
        at_fold.append(_cv_metrics(y_test, at_pred))

    def _agg(folds: list[dict]) -> dict:
        out = {
            "mae": float(np.mean([f["mae"] for f in folds])),
            "rmse": float(np.mean([f["rmse"] for f in folds])),
            "r_squared": float(np.mean([f["r_squared"] for f in folds])),
            "mae_std": float(np.std([f["mae"] for f in folds])),
            "rmse_std": float(np.std([f["rmse"] for f in folds])),
            "r_squared_std": float(np.std([f["r_squared"] for f in folds])),
            "cv_folds": 5,
        }
        return out

    results = {
        "global_median": _agg(gm_fold),
        "area_type_median": _agg(at_fold),
    }
    for name, m in results.items():
        log.info("baseline_evaluated", baseline=name, **{k: round(v, 4) for k, v in m.items()})
    return results
