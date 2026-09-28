"""ISOLATED experiment: target leakage through the per-sqm density features.

The target is ``gross_yield_lt = median rent / median sale price * 100`` and two
features are built from the same two medians in the same row:
``price_per_sqm = median price / sale area`` and ``rent_per_sqm = median rent /
rent area``. Together with ``floor_area_median_sqm`` they let a model rebuild most
of the target arithmetically, so cross-validated accuracy may measure that
identity rather than learned market structure.

Read-only and side-effect-free: NO database writes, NO model persistence. Every
variant uses the production model configuration and identical 5-fold CV:

  A  production features (same-row per-sqm, as trained and reported)
  B  both per-sqm features removed
  C  only rent_per_sqm removed
  D  per-sqm features lagged to the segment's previous observed year, which is
     leakage-free and mirrors serving (``ml.predict`` reads the latest known year)
  E  no model: predict the segment's previous observed yield (persistence)

D and E need a previous year, so A and B are also reported on that subset.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m ml.experiments.run_leakage_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from ml.encoding import encode
from ml.persist import active_run_row, feature_matrix_rows

TARGET = "gross_yield_lt"
SEGMENT = ["community", "property_type", "bedrooms"]
DENSITY = ["price_per_sqm", "rent_per_sqm"]

XGB_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.08,
    "max_depth": 4,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "reg_alpha": 0.0,
}


def _models() -> dict:
    import xgboost as xgb

    return {
        "xgboost": lambda: xgb.XGBRegressor(
            random_state=42, verbosity=0, n_jobs=-1, **XGB_PARAMS
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }


def _cv(make, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s, maes = [], []
    for tr, te in kf.split(X):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(y[te], pred))
    return float(np.mean(r2s)), float(np.mean(maes))


def _with_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([*SEGMENT, "year"]).reset_index(drop=True)
    g = df.groupby(SEGMENT, sort=False)
    for c in [*DENSITY, TARGET, "year"]:
        df[f"prev_{c}"] = g[c].shift(1)
    return df


def main() -> None:
    run = active_run_row(phase=1)
    version = run.get("feature_version") if run else None
    df = pd.DataFrame(feature_matrix_rows(version))
    for c in [TARGET, *DENSITY, "floor_area_median_sqm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    df = _with_lags(df)
    y_all = df[TARGET].astype(float).values
    print(f"rows: {len(df)}  (active run {run['run_name'] if run else '-'})")

    ratio = 100 * df["rent_per_sqm"] / df["price_per_sqm"]
    ok = ratio.notna() & np.isfinite(ratio) & (df["price_per_sqm"] > 0)
    print("\nArithmetic reconstruction, no model:")
    print(
        f"  100 * rent_per_sqm / price_per_sqm vs target: "
        f"R2 = {r2_score(y_all[ok], ratio[ok]):.4f}  "
        f"corr = {np.corrcoef(y_all[ok], ratio[ok])[0, 1]:.4f}  (n={int(ok.sum())})"
    )

    X_full = encode(df)
    variants_all = {
        "A  production features": X_full,
        "B  without both per-sqm": X_full.drop(columns=DENSITY),
        "C  without rent_per_sqm": X_full.drop(columns=["rent_per_sqm"]),
    }

    sub = df["prev_year"].notna().to_numpy()
    df_sub = df[sub].reset_index(drop=True)
    y_sub = y_all[sub]
    lagged = df_sub.copy()
    for c in DENSITY:
        lagged[c] = lagged[f"prev_{c}"]
    X_sub = X_full[sub].reset_index(drop=True)
    X_lag = X_sub.copy()
    for c in DENSITY:
        X_lag[c] = pd.to_numeric(lagged[c], errors="coerce").fillna(0.0).values
    variants_sub = {
        "A  production features": X_sub,
        "B  without both per-sqm": X_sub.drop(columns=DENSITY),
        "D  per-sqm lagged one year": X_lag,
    }

    for label, variants, y in (
        (f"All rows (n={len(df)})", variants_all, y_all),
        (f"Rows with a previous year (n={len(df_sub)})", variants_sub, y_sub),
    ):
        print(f"\n{label}")
        print(f"  {'variant':32s} {'model':8s} {'R2':>7s} {'MAE':>7s}")
        for name, X in variants.items():
            for mname, make in _models().items():
                r2, mae = _cv(make, X, y)
                print(f"  {name:32s} {mname:8s} {r2:7.4f} {mae:7.4f}")

    prev = df_sub[f"prev_{TARGET}"].astype(float).values
    print("\n  E  persistence (last observed yield), no model: "
          f"R2 = {r2_score(y_sub, prev):.4f}  MAE = {mean_absolute_error(y_sub, prev):.4f}")
    gap = (df_sub["year"] - df_sub["prev_year"]).value_counts().sort_index()
    print("  years back to the previous observation: "
          + ", ".join(f"{int(k)}y={int(v)}" for k, v in gap.items()))


if __name__ == "__main__":
    main()
