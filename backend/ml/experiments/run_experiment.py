"""ISOLATED experiment: V1 (live) vs V2 (bedroom grain + per-sqm) feature sets.

Read-only and side-effect-free - NO database writes, NO model persistence, NO
run activation. It rebuilds each feature matrix, evaluates the naive baselines
and the three models (RF hyperparameter-tuned) under identical 5-fold CV, and
also tries a log-transformed target for each. Prints a side-by-side comparison so
the user can decide whether to adopt V2.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m ml.experiments.run_experiment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from db.queries.feature_matrix_dld import build_feature_matrix as build_v1
from ml.experiments.feature_matrix_v2 import build_feature_matrix_v2 as build_v2
from ml.train import _tune_rf, _models

log = structlog.get_logger(__name__)

TARGET = "gross_yield_lt"

_CAT = ["property_type", "community", "nearest_metro"]
_NUM_V1 = ["year", "floor_area_median_sqm", "bedrooms_median", "parking_rate", "n_sales", "n_rents"]
_NUM_V2 = ["year", "bedrooms", "floor_area_median_sqm", "price_per_sqm", "rent_per_sqm",
           "parking_rate", "n_sales", "n_rents"]

_IMPUTE = {"bedrooms_median": 1.0, "bedrooms": 1.0, "floor_area_median_sqm": 0.0,
           "price_per_sqm": 0.0, "rent_per_sqm": 0.0, "parking_rate": 0.0}


def _encode(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    present_num = [c for c in numeric if c in df.columns]
    present_cat = [c for c in _CAT if c in df.columns]
    X = df[present_num].copy()
    for c in present_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(_IMPUTE.get(c, 0))
    if present_cat:
        dummies = pd.get_dummies(df[present_cat].astype("object"), prefix=present_cat)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X


def _cv_model(model, X, y, kf, log_target: bool) -> dict:
    maes, rmses, r2s = [], [], []
    for tr, te in kf.split(X):
        yt = np.log1p(y[tr]) if log_target else y[tr]
        model.fit(X.iloc[tr], yt)
        pred = model.predict(X.iloc[te])
        if log_target:
            pred = np.expm1(pred)
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        r2s.append(r2_score(y[te], pred))
    return {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
            "r_squared": float(np.mean(r2s))}


def _baselines(df, y, keys, kf) -> dict:
    gm, at = [], []
    for tr, te in kf.split(df):
        gmed = float(np.median(y[tr]))
        buckets: dict = {}
        for i in tr:
            buckets.setdefault(keys[i], []).append(y[i])
        grp = {k: float(np.median(v)) for k, v in buckets.items()}
        gm.append(r2_score(y[te], [gmed] * len(te)))
        at.append(r2_score(y[te], [grp.get(keys[i], gmed) for i in te]))
    return {"global_median_r2": float(np.mean(gm)),
            "area_type_median_r2": float(np.mean(at))}


def _evaluate(label: str, rows: list[dict], numeric: list[str]) -> None:
    df = pd.DataFrame(rows)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = df[TARGET].astype(float).values
    X = _encode(df, numeric)
    keys = list(zip(df["community"], df["property_type"]))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'='*70}\n{label}: {len(df)} rows, {X.shape[1]} features\n{'='*70}")
    base = _baselines(df, y, keys, kf)
    print(f"  baseline area+type median:  R2={base['area_type_median_r2']:.3f}")

    rf_tuned, rf_params = _tune_rf(X, y, kf)
    models = _models(rf_tuned)
    print(f"  {'model':<12} {'MAE':>7} {'RMSE':>7} {'R2':>7}   {'R2(log)':>8}")
    for name, model in models.items():
        raw = _cv_model(model, X, y, kf, log_target=False)
        logv = _cv_model(model, X, y, kf, log_target=True)
        print(f"  {name:<12} {raw['mae']:>7.3f} {raw['rmse']:>7.3f} "
              f"{raw['r_squared']:>7.3f}   {logv['r_squared']:>8.3f}")
    print(f"  RF tuned params: {rf_params}")


def main() -> None:
    print("Building feature matrices (read-only; no DB writes)...")
    v1 = build_v1(feature_version=1)
    v2 = build_v2(feature_version=2)
    _evaluate("V1 (live: area x type x year)", v1, _NUM_V1)
    _evaluate("V2 (experimental: + bedrooms + per-sqm)", v2, _NUM_V2)
    print("\nDone. No changes persisted - the live phase1_v1_rf model is untouched.")


if __name__ == "__main__":
    main()
