"""Held-out evaluation of the yield model (dissertation Results/Discussion).

The production trainer (``ml/train.py``) trains on ALL rows - correct for the
deployed model. But a headline accuracy claim must be measured on data the model
never saw *at any stage* (not even for model selection or tuning). This module
provides that honest generalisation number:

    1. Build the feature matrix, then hold out a random **20 % test set FIRST**
       (seeded, stratified by property type), before anything else touches it.
    2. On the **80 % train set only**: hyperparameter-tune RF + XGBoost, run
       5-fold CV across LightGBM / RF / XGBoost, and pick the winner by CV R².
    3. Refit the winner on the full 80 %, then evaluate **once** on the untouched
       20 % → reported MAE / RMSE / R². Also report the train-vs-test gap as an
       overfitting check.
    4. Residual / error analysis on the test set (error by property type,
       bedrooms, price band, yield band) for the Discussion chapter.

Read-only w.r.t. the DB and the production artefact - it neither persists nor
activates anything. Run:
    cd backend && .venv\\Scripts\\python.exe -m ml.evaluate_holdout
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from ml.encoding import encode
from ml.train import TARGET, _load_rows, _models, _tune_rf, _tune_xgb

log = structlog.get_logger(__name__)

TEST_SIZE = 0.20
SEED = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r_squared": float(r2_score(y_true, y_pred)),
    }


def _select_on_train(X_tr: pd.DataFrame, y_tr) -> tuple[str, object, dict]:
    """Tune + 5-fold CV all three models on the TRAIN set; return the CV winner
    (by R²), refit on the full train set, plus the CV table."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_est, _ = _tune_rf(X_tr, y_tr, kf)
    xgb_est, _ = _tune_xgb(X_tr, y_tr, kf)

    cv: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in _models(rf_est, xgb_est).items():
        r2s, maes = [], []
        for tr, te in kf.split(X_tr):
            model.fit(X_tr.iloc[tr], y_tr[tr])
            p = model.predict(X_tr.iloc[te])
            r2s.append(r2_score(y_tr[te], p))
            maes.append(mean_absolute_error(y_tr[te], p))
        cv[name] = {"cv_r2": float(np.mean(r2s)), "cv_mae": float(np.mean(maes))}
        model.fit(X_tr, y_tr)
        fitted[name] = model
        log.info("holdout_cv", model=name, **{k: round(v, 4) for k, v in cv[name].items()})

    best = max(cv, key=lambda k: cv[k]["cv_r2"])
    return best, fitted[best], cv


def _quantile_band(series: pd.Series, extremes: tuple[str, str]) -> pd.Series:
    """Label a numeric series into up to 4 quantile bands, tie-safe.

    Uses ``labels=False`` so the number of labels always equals the number of bins
    that actually survive ``duplicates="drop"`` (a fixed label list raises when ties
    collapse an edge). The integer codes are then mapped to ``Q{i} <name>`` where
    the lowest/highest carry the human ``extremes`` words. NaN codes (all-equal
    input) become a single "All" band.
    """
    codes = pd.qcut(series, 4, labels=False, duplicates="drop")
    n = int(codes.max()) + 1 if codes.notna().any() else 0
    if n <= 1:
        return pd.Series(["All"] * len(series), index=series.index)

    def name(i: int) -> str:
        if i == 0:
            return f"Q1 {extremes[0]}"
        if i == n - 1:
            return f"Q{n} {extremes[1]}"
        return f"Q{i + 1}"

    return codes.map(lambda c: name(int(c)) if pd.notna(c) else "All")


def _residual_analysis(df_test: pd.DataFrame, y_true, y_pred) -> dict:
    """Mean absolute error sliced by segment - where the model errs."""
    d = df_test.copy()
    d["_abs_err"] = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    d["_price_band"] = _quantile_band(
        d["price_per_sqm"].astype(float), ("cheapest", "priciest")
    )
    d["_yield_band"] = _quantile_band(
        d["gross_yield_lt"].astype(float), ("lowest", "highest")
    )

    def by(col: str) -> dict:
        g = d.groupby(col, observed=True)["_abs_err"].agg(["mean", "count"])
        return {str(k): {"mae": round(float(v["mean"]), 3), "n": int(v["count"])}
                for k, v in g.iterrows()}

    return {
        "by_property_type": by("property_type"),
        "by_bedrooms": by("bedrooms"),
        "by_price_band": by("_price_band"),
        "by_yield_band": by("_yield_band"),
        "worst_10_abs_error": [
            {
                "community": r["community"], "type": r["property_type"],
                "beds": int(r["bedrooms"]), "year": int(r["year"]),
                "actual": round(float(r["gross_yield_lt"]), 2),
                "abs_err": round(float(r["_abs_err"]), 2),
            }
            for _, r in d.nlargest(10, "_abs_err").iterrows()
        ],
    }


def run_holdout(feature_version: int | None = None, target: str = TARGET) -> dict:
    df = pd.DataFrame(_load_rows(feature_version)).reset_index(drop=True)
    df = df[df[target].notna()].reset_index(drop=True)

    X_all = encode(df).reset_index(drop=True)
    y_all = df[target].astype(float).values
    strat = df["property_type"] if df["property_type"].value_counts().min() >= 2 else None

    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=strat
    )
    X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
    df_te = df.iloc[te_idx].reset_index(drop=True)

    log.info("holdout_split", train=len(tr_idx), test=len(te_idx), test_size=TEST_SIZE)

    best_name, best_model, cv = _select_on_train(X_tr, y_tr)

    test_metrics = _metrics(y_te, best_model.predict(X_te))
    train_metrics = _metrics(y_tr, best_model.predict(X_tr))

    residuals = _residual_analysis(df_te, y_te, best_model.predict(X_te))

    log.info("holdout_result", model=best_name,
             **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()})
    return {
        "selected_model": best_name,
        "cv_on_train": cv,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_gap_r2": round(train_metrics["r_squared"] - test_metrics["r_squared"], 4),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "residuals": residuals,
    }


def _print_report(res: dict) -> None:
    print(f"\n{'='*66}\nHELD-OUT EVALUATION  (test set held out before tuning/selection)\n{'='*66}")
    print(f"train rows: {res['n_train']}   test rows: {res['n_test']}   (seed {SEED})")
    print("\nCV on TRAIN (model selection):")
    for m, v in res["cv_on_train"].items():
        print(f"  {m:<10} cv_R2={v['cv_r2']:.3f}  cv_MAE={v['cv_mae']:.3f}")
    print(f"\nSelected (CV winner): {res['selected_model']}")
    tr, te = res["train_metrics"], res["test_metrics"]
    print(f"\n{'':12}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    print(f"  {'train':<10}{tr['mae']:>8.3f}{tr['rmse']:>8.3f}{tr['r_squared']:>8.3f}")
    print(f"  {'TEST':<10}{te['mae']:>8.3f}{te['rmse']:>8.3f}{te['r_squared']:>8.3f}  <-- reported")
    print(f"  overfit gap (train R2 - test R2): {res['overfit_gap_r2']}")
    r = res["residuals"]
    print("\nTest-set MAE by property type:")
    for k, v in r["by_property_type"].items():
        print(f"  {k:<16} MAE={v['mae']:.3f}  n={v['n']}")
    print("Test-set MAE by bedrooms:")
    for k, v in sorted(r["by_bedrooms"].items()):
        print(f"  {k:>2} bed          MAE={v['mae']:.3f}  n={v['n']}")
    print("Test-set MAE by price/sqm band:")
    for k, v in r["by_price_band"].items():
        print(f"  {k:<16} MAE={v['mae']:.3f}  n={v['n']}")
    print("Test-set MAE by yield band:")
    for k, v in r["by_yield_band"].items():
        print(f"  {k:<16} MAE={v['mae']:.3f}  n={v['n']}")


if __name__ == "__main__":
    _print_report(run_holdout())
