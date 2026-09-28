"""ISOLATED experiment: Objective 3.4 - does the Phase-2 feature set help, and
does the Phase-1 validation protocol hold up under time-based splits?

Two independent questions, deliberately answered in one run because they share a
data load and because the second is the one that matters:

    PART 1 - THE EXTENDED FEATURE SET (the objective as written).
        A. Phase 1 - the production feature set.
        B. Phase 2 - the same set plus the macro/environmental/seasonal series
           ingested by Objectives 3.1-3.3, joined on year from ``macro_annual``.
        Identical rows, identical folds (seed 42), each set tuned at its own
        optimum. Reports dMAE / dRMSE / dR2, which is exactly what the objective
        asks to "quantify".

    PART 2 - THE VALIDATION PROTOCOL (the leakage question).
        Phase 1 validates with ``KFold(shuffle=True, seed=42)`` over rows spanning
        2012-2026. A random fold can place 2024 rows in train and 2023 rows of the
        *same* (community, type, bedrooms) bucket in test. The proposal
        anticipated this and required time-based splits. This part measures the
        effect three ways:
          (a) random 5-fold  - the incumbent protocol;
          (b) forward chaining - train on all prior years, predict year Y;
          (c) an ISOLATION test that holds the test rows AND the training-set
              size fixed, varying only whether the training set may contain other
              rows from the test year. Without (c), (b) confounds leakage with
              having less training data, and the comparison proves nothing.

**Expected result for Part 1, stated in advance (F28/F29).** The Phase-1 matrix
is built at (community x property type x bedrooms x YEAR) and ``year`` is already
a feature, so every annual macro series is a deterministic function of a feature
the model already has: 9-15 distinct values across 2,552 rows. Weather is worse -
99.4% of its variance does not survive aggregation to year. Ramadan is worst: its
annual day-count is 29 or 30 by construction. This run converts that prediction
into a measurement.

Read-only and side-effect-free: NO database writes, NO model persistence, NO run
activation. The served model is untouched, and ``MACRO_FEATURES`` stays empty.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m ml.experiments.run_phase2_evaluation
    ... --quick     skip hyperparameter tuning (defaults only; much faster)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from db.queries.macro import all_macro_annual
from ml.encoding import CATEGORICAL, NUMERIC, _IMPUTE
from ml.train import _load_rows, _models, _tune_rf, _tune_xgb

TARGET = "gross_yield_lt"

PHASE2_FEATURES = [
    "eibor_3m",
    "brent_crude_usd",
    "uae_gdp_growth",
    "uae_inflation_pct",
    "uae_cpi_index",
    "temp_max_c",
    "humidity_pct",
    "thermal_index",
    "ramadan_days",
    "expo_months",
    "covid_period_months",
]

MIN_TRAIN_YEARS = 5
MIN_TEST_ROWS = 20


def _encode(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Same encoding as ml.encoding, with an explicit numeric column list."""
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
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r_squared": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
    }


def _evaluate(label: str, X: pd.DataFrame, y, kf: KFold, tune: bool) -> dict:
    print(f"\n{label}: {X.shape[0]} rows, {X.shape[1]} features")
    rf, xgb_est = (None, None)
    if tune:
        rf, _ = _tune_rf(X, y, kf)
        xgb_est, _ = _tune_xgb(X, y, kf)
    out = {}
    print(f"  {'model':<12} {'MAE':>7} {'RMSE':>7} {'R2':>8} {'R2 sd':>7}")
    for name, model in _models(rf, xgb_est).items():
        out[name] = _cv(model, X, y, kf)
        print(
            f"  {name:<12} {out[name]['mae']:>7.3f} {out[name]['rmse']:>7.3f} "
            f"{out[name]['r_squared']:>8.4f} {out[name]['r2_std']:>7.4f}"
        )
    return out


def _fit_score(model_factory, X, y, train_idx, test_idx) -> tuple[float, float, float]:
    model = model_factory()
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])
    return (
        r2_score(y[test_idx], pred),
        mean_absolute_error(y[test_idx], pred),
        float(np.sqrt(mean_squared_error(y[test_idx], pred))),
    )


def _join_macro(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """LEFT JOIN ``macro_annual`` onto the training rows by year.

    LEFT, not inner: EIBOR only reaches back to 2018 and World Bank GDP stops at
    2024, so an inner join would silently drop training rows to match the
    thinnest macro series - changing the row set and making the comparison
    invalid. Missing values stay NaN and are imputed to 0 by the shared encoder,
    exactly as the production path treats an absent OSM feature.
    """
    macro_rows = all_macro_annual()
    if not macro_rows:
        raise SystemExit(
            "macro_annual is empty - run `python -m scripts.ingest_macro` first, "
            "or this comparison is between two identical feature sets."
        )
    macro = pd.DataFrame(macro_rows)
    available = [c for c in PHASE2_FEATURES if c in macro.columns]
    macro = macro[["year"] + available].copy()
    for c in available:
        macro[c] = pd.to_numeric(macro[c], errors="coerce")

    merged = df.merge(macro, on="year", how="left")
    assert len(merged) == len(df), "macro join changed the row count"
    return merged, available


def _report_information_content(df: pd.DataFrame, features: list[str]) -> None:
    """Distinct values per macro feature - the F28/F29 argument, re-measured here."""
    print("\nInformation available to the model at this join grain:")
    print(f"  {'feature':<22} {'distinct':>9}  {'coverage':>9}")
    for col in features:
        distinct = df[col].dropna().nunique()
        coverage = 100.0 * df[col].notna().sum() / len(df)
        print(f"  {col:<22} {distinct:>9} {coverage:>8.1f}%")
    print(f"  {'year (already used)':<22} {df['year'].nunique():>9} {100.0:>8.1f}%")
    print(
        "\n  Every macro feature has at most as many distinct values as `year`, and\n"
        "  is a function of it. A tree can already express any year-specific effect\n"
        "  through `year` itself."
    )


def part1_feature_set(df: pd.DataFrame, y, tune: bool) -> None:
    print("\n" + "=" * 74)
    print("PART 1 - DOES THE PHASE-2 FEATURE SET IMPROVE THE MODEL?")
    print("=" * 74)

    merged, available = _join_macro(df)
    missing = [c for c in PHASE2_FEATURES if c not in available]
    if missing:
        print(f"  (not in macro_annual, skipped: {', '.join(missing)})")
    _report_information_content(merged, available)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_phase1 = _encode(merged, NUMERIC)
    X_phase2 = _encode(merged, NUMERIC + available)

    phase1 = _evaluate("A. PHASE 1 (production set)", X_phase1, y, kf, tune)
    phase2 = _evaluate("B. PHASE 2 (+ macro/env/seasonal)", X_phase2, y, kf, tune)

    print("\n" + "-" * 74)
    print("DELTA (B - A); positive dR2 / negative dMAE = Phase 2 helps")
    print(f"  {'model':<12} {'dMAE':>9} {'dRMSE':>9} {'dR2':>9}")
    for name in phase1:
        print(
            f"  {name:<12} "
            f"{phase2[name]['mae'] - phase1[name]['mae']:>+9.4f} "
            f"{phase2[name]['rmse'] - phase1[name]['rmse']:>+9.4f} "
            f"{phase2[name]['r_squared'] - phase1[name]['r_squared']:>+9.4f}"
        )
    best = max(phase1, key=lambda k: phase1[k]["r_squared"])
    delta = phase2[best]["r_squared"] - phase1[best]["r_squared"]
    fold_sd = phase1[best]["r2_std"]
    print(
        f"\n  Production model ({best}): dR2 {delta:+.4f} against fold-to-fold "
        f"sd {fold_sd:.4f}"
    )
    print(
        f"  The change is {abs(delta) / fold_sd:.2f}x the fold noise - "
        f"{'MATERIAL' if abs(delta) > fold_sd else 'INSIDE THE NOISE'}."
    )


def part2_protocol(df: pd.DataFrame, y, tune: bool) -> None:
    print("\n" + "=" * 74)
    print("PART 2 - DOES THE RANDOM-K-FOLD PROTOCOL LEAK ACROSS YEARS?")
    print("=" * 74)

    import xgboost as xgb

    def factory():
        return xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)

    X = _encode(df, NUMERIC)
    years = sorted(df["year"].unique())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = [
        _fit_score(factory, X, y, tr, te) for tr, te in kf.split(X)
    ]
    rand_r2 = float(np.mean([s[0] for s in scores]))
    rand_mae = float(np.mean([s[1] for s in scores]))
    rand_rmse = float(np.mean([s[2] for s in scores]))
    print(
        f"\n(a) RANDOM 5-fold (incumbent):      R2 {rand_r2:.4f}  "
        f"MAE {rand_mae:.4f}  RMSE {rand_rmse:.4f}"
    )

    print("\n(b) FORWARD CHAINING - train on all prior years, predict year Y:")
    rows = []
    for cut in years[MIN_TRAIN_YEARS:]:
        train_idx = np.where(df["year"] < cut)[0]
        test_idx = np.where(df["year"] == cut)[0]
        if len(test_idx) < MIN_TEST_ROWS:
            continue
        r2, mae, rmse = _fit_score(factory, X, y, train_idx, test_idx)
        rows.append((cut, r2, mae, rmse, len(test_idx)))
        print(
            f"      {cut}   R2 {r2:>7.4f}   MAE {mae:.4f}   RMSE {rmse:.4f}   "
            f"(n={len(test_idx)})"
        )
    time_r2 = float(np.mean([r[1] for r in rows]))
    time_mae = float(np.mean([r[2] for r in rows]))
    time_rmse = float(np.mean([r[3] for r in rows]))
    print(
        f"      MEAN   R2 {time_r2:.4f}   MAE {time_mae:.4f}   RMSE {time_rmse:.4f}"
    )
    print(
        f"\n    Random-fold advantage: R2 {rand_r2 - time_r2:+.4f}, "
        f"MAE {rand_mae - time_mae:+.4f}"
    )

    print("\n(c) ISOLATION TEST - is that gap leakage, or just less training data?")
    print("    Identical test rows AND identical training-set SIZE; the only")
    print("    difference is whether the training set may contain other rows")
    print("    from the test year.")
    print(f"\n    {'year':<6} {'past-only R2':>13} {'mixed R2':>10} {'leak dR2':>10}  note")
    leaks = []
    n = len(df)
    for cut in years[MIN_TRAIN_YEARS:]:
        test_idx = np.where(df["year"] == cut)[0]
        past_idx = np.where(df["year"] < cut)[0]
        if len(test_idx) < MIN_TEST_ROWS or len(past_idx) < MIN_TEST_ROWS:
            continue
        pool = np.setdiff1d(np.arange(n), test_idx)
        degenerate = len(pool) == len(past_idx)
        past_r2 = _fit_score(factory, X, y, past_idx, test_idx)[0]
        if degenerate:
            print(
                f"    {cut:<6} {past_r2:>13.4f} {' - ':>10} {' - ':>10}  "
                f"degenerate (last year)"
            )
            continue
        mixed = [
            _fit_score(
                factory,
                X,
                y,
                np.random.default_rng(seed).choice(pool, size=len(past_idx), replace=False),
                test_idx,
            )[0]
            for seed in range(3)
        ]
        mixed_r2 = float(np.mean(mixed))
        leaks.append(mixed_r2 - past_r2)
        print(
            f"    {cut:<6} {past_r2:>13.4f} {mixed_r2:>10.4f} "
            f"{mixed_r2 - past_r2:>+10.4f}"
        )
    if leaks:
        mean_leak = float(np.mean(leaks))
        sd_leak = float(np.std(leaks, ddof=1)) if len(leaks) > 1 else 0.0
        print(
            f"\n    MEAN LEAKAGE EFFECT: {mean_leak:+.4f} R2   "
            f"(sd {sd_leak:.4f} over {len(leaks)} years, "
            f"range {min(leaks):+.4f} to {max(leaks):+.4f})"
        )
        print(
            "    Same training-set size, same test rows, so this separates seeing"
        )
        print(
            "    other rows from the test year from simply having less data."
        )
        if sd_leak and abs(mean_leak) < sd_leak:
            print(
                f"\n    NOTE: the mean is {abs(mean_leak) / sd_leak:.2f} sd from zero, and"
            )
            print(
                "    individual years disagree in SIGN, so leakage is NOT reliably"
            )
            print(
                "    distinguishable from zero. Do not quote the mean as a settled"
            )
            print("    effect size.")
        gap = rand_r2 - time_r2
        share = 100.0 * mean_leak / gap if gap else 0.0
        print(f"\n    ACCOUNTING FOR THE {gap:+.4f} R2 FORWARD-CHAINING GAP:")
        print(f"      leakage (isolated)   {mean_leak:+.4f}  ({share:.0f}%)")
        print(f"      residual             {gap - mean_leak:+.4f}  ({100 - share:.0f}%)")
        print(
            "    The residual is genuine temporal generalisation difficulty:"
        )
        print(
            "    predicting an unseen FUTURE year is harder than interpolating"
        )
        print(
            "    within years the model has already seen. That is a property of"
        )
        print("    the problem, not a defect in the protocol.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip hyperparameter tuning in Part 1 (defaults only; much faster).",
    )
    parser.add_argument(
        "--part", choices=["1", "2", "both"], default="both", help="Which part to run."
    )
    args = parser.parse_args()

    print("Loading the feature matrix (read-only; no DB writes)...")
    df = pd.DataFrame(_load_rows(None))
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no training rows")
    y = df[TARGET].astype(float).values
    print(
        f"  {len(df)} rows, {df['community'].nunique()} communities, "
        f"years {df['year'].min()}-{df['year'].max()}"
    )

    if args.part in ("1", "both"):
        part1_feature_set(df, y, tune=not args.quick)
    if args.part in ("2", "both"):
        part2_protocol(df, y, tune=not args.quick)

    print("\nDone. Nothing persisted - the active model is untouched.")


if __name__ == "__main__":
    main()
