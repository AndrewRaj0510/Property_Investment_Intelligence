"""ISOLATED experiment: does the OpenStreetMap enrichment improve the model?

Answers the question Objective 1.4 actually raises - not "was OSM data ingested"
but "does it carry signal the model didn't already have". The DLD sales table
already contains a ``nearest_metro_en`` *station name* (one-hot encoded), so the
OSM pair must earn its place against a location signal that is partly there:

    A. WITHOUT OSM - the production feature set as of F13.
    B. WITH OSM    - the same set plus nearest_metro_km + poi_density_1km.

Identical rows, identical 5-fold splits (seed 42), identical tuned models, so the
delta is attributable to the two columns and nothing else. Also reports coverage,
because a gain measured over rows that are 20 % enriched means something quite
different from one measured at 95 %.

Read-only and side-effect-free: NO database writes, NO model persistence, NO run
activation. The live model is untouched.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m ml.experiments.run_osm_experiment
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from db.queries.feature_matrix_dld import build_feature_matrix
from ml.encoding import CATEGORICAL, NUMERIC, _IMPUTE
from ml.train import _models, _tune_rf, _tune_xgb

TARGET = "gross_yield_lt"

OSM_FEATURES = ["nearest_metro_km", "poi_density_1km"]
BASE_NUMERIC = [c for c in NUMERIC if c not in OSM_FEATURES]


def _encode(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Same encoding as ml.encoding, but with an explicit numeric column list."""
    present_num = [c for c in numeric if c in df.columns]
    present_cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[present_num].copy()
    for c in present_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(_IMPUTE.get(c, 0))
    if present_cat:
        dummies = pd.get_dummies(df[present_cat].astype("object"), prefix=present_cat)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X


def _cv(model, X: pd.DataFrame, y, kf: KFold) -> dict:
    maes, rmses, r2s = [], [], []
    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        r2s.append(r2_score(y[te], pred))
    return {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
            "r_squared": float(np.mean(r2s))}


def _evaluate(label: str, df: pd.DataFrame, y, numeric: list[str], kf: KFold) -> dict:
    X = _encode(df, numeric)
    print(f"\n{label}: {len(df)} rows, {X.shape[1]} features")
    rf, _ = _tune_rf(X, y, kf)
    xgb_est, _ = _tune_xgb(X, y, kf)
    out = {}
    print(f"  {'model':<12} {'MAE':>7} {'RMSE':>7} {'R2':>8}")
    for name, model in _models(rf, xgb_est).items():
        out[name] = _cv(model, X, y, kf)
        print(f"  {name:<12} {out[name]['mae']:>7.3f} {out[name]['rmse']:>7.3f} "
              f"{out[name]['r_squared']:>8.3f}")
    return out


def main() -> None:
    print("Building the feature matrix (read-only; no DB writes)...")
    rows = build_feature_matrix(feature_version=1)
    df = pd.DataFrame(rows)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no training rows")

    enriched = int(df["poi_density_1km"].notna().sum())
    pct = 100.0 * enriched / len(df)
    communities = df["community"].nunique()
    enriched_comms = df.loc[df["poi_density_1km"].notna(), "community"].nunique()
    print(f"\nOSM coverage: {enriched}/{len(df)} training rows ({pct:.1f}%), "
          f"{enriched_comms}/{communities} communities")
    if enriched == 0:
        raise SystemExit(
            "No community has OSM features yet - run "
            "`python -m scripts.refresh_osm_features` first, or this comparison "
            "is between two identical feature sets."
        )

    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 70)
    without = _evaluate("A. WITHOUT OSM", df, y, BASE_NUMERIC, kf)
    with_osm = _evaluate("B. WITH OSM   ", df, y, NUMERIC, kf)

    print("\n" + "=" * 70)
    print("DELTA (B - A); positive R2 / negative MAE = OSM helps")
    print(f"  {'model':<12} {'dMAE':>8} {'dRMSE':>8} {'dR2':>8}")
    for name in without:
        d_mae = with_osm[name]["mae"] - without[name]["mae"]
        d_rmse = with_osm[name]["rmse"] - without[name]["rmse"]
        d_r2 = with_osm[name]["r_squared"] - without[name]["r_squared"]
        print(f"  {name:<12} {d_mae:>+8.4f} {d_rmse:>+8.4f} {d_r2:>+8.4f}")

    best_a = max(without, key=lambda k: without[k]["r_squared"])
    best_b = max(with_osm, key=lambda k: with_osm[k]["r_squared"])
    print(f"\nBest without OSM: {best_a} R2={without[best_a]['r_squared']:.4f}")
    print(f"Best with OSM:    {best_b} R2={with_osm[best_b]['r_squared']:.4f}")
    print("\nDone. Nothing persisted - the active model is untouched.")


if __name__ == "__main__":
    main()
