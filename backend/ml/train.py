"""Model training + cross-validation (M3B-01).

Trains LightGBM, Random Forest, and XGBoost on the feature matrix with 5-fold
K-fold CV, records MAE/RMSE/R², and returns the best model (chosen empirically by
CV R²) plus a metrics dict. Random Forest is additionally hyperparameter-tuned
via randomized search before the comparison.

Model selection is DATA-DRIVEN: the proposal specified building LightGBM and then
comparing it against alternatives, so the production model is whichever wins the
cross-validated comparison (Random Forest, as of 2026-07-24) rather than a
hard-coded choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from ml.encoding import encode
from ml.persist import feature_matrix_rows

log = structlog.get_logger(__name__)

TARGET = "gross_yield_lt"

PREFERRED_MODEL = "rf"


def _load_rows(feature_version: int | None) -> list[dict]:
    """Fetch the persisted feature matrix; fall back to building it live.

    Normal flow: the training entrypoint has already materialised the
    (area, type, bedrooms, year) matrix into the ``feature_matrix`` table for
    this version, and training reads that exact set so SHAP analysis (which reads
    the table) sees identical rows.

    Fallback: if the table read yields nothing or is unreachable, build the matrix
    directly from the raw DLD tables. This keeps standalone training runnable.

    The read goes through ``ml.persist`` (psycopg). It previously went through
    ``db.queries.features.feature_matrix``, which uses the Supabase ORM client -
    unconfigured in this deployment, so that call could only ever raise and fall
    through, quietly rebuilding the matrix on every run instead of training on
    the materialised rows.
    """
    from db.queries.feature_matrix_dld import build_feature_matrix

    try:
        rows = feature_matrix_rows(feature_version)
        if rows:
            return rows
        log.warning("feature_matrix_table_empty_building_live", version=feature_version)
    except Exception as exc:  # noqa: BLE001
        log.warning("feature_matrix_table_unavailable_building_live", error=str(exc))
    return build_feature_matrix(feature_version or 1)


_RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 600, 800],
    "max_depth": [None, 8, 12, 16, 24],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}


def _tune_rf(X: pd.DataFrame, y, kf: KFold) -> tuple[RandomForestRegressor, dict]:
    """Hyperparameter-tune Random Forest via randomized search (CV = ``kf``).

    Optimises negative MAE (the primary reporting metric). Returns an *unfitted*
    estimator configured with the best params (so the caller cross-validates it
    on the same folds as the other models for a fair comparison) plus the chosen
    params for the run record.
    """
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=_RF_PARAM_DISTRIBUTIONS,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=kf,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best = search.best_params_
    log.info("rf_tuned", best_params=best, best_cv_mae=round(-search.best_score_, 4))
    return (
        RandomForestRegressor(random_state=42, n_jobs=-1, **best),
        best,
    )


_XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 700, 900],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.5, 1.0],
}


def _tune_xgb(X: pd.DataFrame, y, kf: KFold):
    """Hyperparameter-tune XGBoost via randomized search (CV = ``kf``).

    Returns an *unfitted* estimator with the best params (so the caller CVs it on
    the same folds as the others) plus the chosen params for the run record.
    """
    import xgboost as xgb

    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        param_distributions=_XGB_PARAM_DISTRIBUTIONS,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=kf,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best = search.best_params_
    log.info("xgb_tuned", best_params=best, best_cv_mae=round(-search.best_score_, 4))
    return xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1, **best), best


def _models(rf_estimator=None, xgb_estimator=None):
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbose=-1,
        ),
        "rf": (
            rf_estimator
            if rf_estimator is not None
            else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        ),
        "xgboost": (
            xgb_estimator
            if xgb_estimator is not None
            else xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5, verbosity=0
            )
        ),
    }


def train_and_evaluate(
    phase: int = 1,
    feature_version: int | None = None,
    target: str = TARGET,
    tune_rf: bool = True,
    tune_xgb: bool = True,
    select_by: str = "r_squared",
) -> dict:
    """Compare models under 5-fold CV and return the empirically-best one.

    ``tune_rf`` / ``tune_xgb`` - hyperparameter-tune RF / XGBoost before the
                    comparison (randomized search on the same folds).
    ``select_by`` - CV metric that decides the winner ("r_squared" → highest,
                    "mae"/"rmse" → lowest). The winning refit model is the
                    served artefact.
    """
    rows = _load_rows(feature_version)
    df = pd.DataFrame(rows)
    if df.empty or target not in df.columns:
        raise ValueError(f"no feature rows for target {target!r}")
    df = df[df[target].notna()].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"insufficient training rows: {len(df)}")

    macro_used: list[str] = []
    if phase == 2:
        from ml.macro_features import join_macro_annual

        df, macro_used = join_macro_annual(df)
        if not macro_used:
            raise ValueError(
                "phase 2 requested but no macro_annual data is available; "
                "run `python -m scripts.ingest_macro` first."
            )

    X = encode(df)
    y = df[target].astype(float).values
    feature_names = list(X.columns)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rf_estimator, rf_best_params = None, None
    if tune_rf:
        rf_estimator, rf_best_params = _tune_rf(X, y, kf)
    xgb_estimator, xgb_best_params = None, None
    if tune_xgb:
        xgb_estimator, xgb_best_params = _tune_xgb(X, y, kf)

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in _models(rf_estimator, xgb_estimator).items():
        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in kf.split(X, y):
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y[test_idx], pred))
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
            r2s.append(r2_score(y[test_idx], pred))
        results[name] = {
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "r_squared": float(np.mean(r2s)),
            "mae_std": float(np.std(maes)),
            "rmse_std": float(np.std(rmses)),
            "r_squared_std": float(np.std(r2s)),
            "cv_folds": n_splits,
            "per_fold": {
                "mae": [round(float(v), 4) for v in maes],
                "rmse": [round(float(v), 4) for v in rmses],
                "r_squared": [round(float(v), 4) for v in r2s],
            },
        }
        model.fit(X, y)
        fitted[name] = model
        log.info("model_trained", model=name, **results[name])

    if select_by == "r_squared":
        best_name = max(results, key=lambda k: results[k]["r_squared"])
    else:
        best_name = min(results, key=lambda k: results[k][select_by])
    log.info(
        "model_selected",
        selected=best_name,
        by=select_by,
        value=round(results[best_name][select_by], 4),
    )

    return {
        "model": fitted[best_name],
        "model_type": best_name,
        "feature_names": feature_names,
        "metrics": results,
        "best_metrics": results[best_name],
        "rf_best_params": rf_best_params,
        "xgb_best_params": xgb_best_params,
        "X": X,
        "y": y,
        "target": target,
        "phase": phase,
        "macro_features_used": macro_used,
    }
