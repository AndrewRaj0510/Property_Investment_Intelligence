"""Phase-2 macro/environmental feature set - the single source of truth.

Phase 1 is the baseline model: property and locational features only. Phase 2 is
the baseline **plus** the external series ingested by Objectives 3.1-3.3, joined
onto the training rows by ``year`` from ``macro_annual``. Comparing the two is
the point of Aim 3 - it is how this project measures whether external factors
affect predicted yield.

This module exists so that train time and serve time cannot disagree. The same
``MACRO_FEATURES`` list drives:

  * the training join (``ml.train.train_and_evaluate`` when ``phase == 2``),
  * the inference join (``ml.predict._build_row`` for a phase-2 bundle),
  * the two-level SHAP grouping (``ml.shap_explainer._group_for``).

**Read this before interpreting a phase-2 result.** The feature matrix is built
at (community x property type x bedrooms x YEAR), and ``year`` is already a
model feature. Every series here is therefore a deterministic function of a
feature the model already has - at most 15 distinct values across 2,552 rows.
The measured consequence (F30) is that adding all of them moves cross-validated
R-squared by **-0.0004**. Weather is worse: 99.4% of its variance does not
survive aggregation to annual grain (F29). ``ramadan_days`` is 29 or 30 by
construction.

So a phase-2 model will still show non-zero SHAP attribution on these features -
trees will happily split on them - but that attribution is **collinear with
``year``**, not independent macroeconomic signal. Attribution share and
predictive contribution are different questions, and only the second one is
answered by the R-squared delta. Do not report the first as though it were the
second.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

MACRO_FEATURES: list[str] = [
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


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2)


def macro_by_year() -> tuple[dict[int, dict], list[str], dict[str, float]]:
    """Return ``({year: {feature: value}}, available_features, impute_defaults)``.

    ``impute_defaults`` is the **median of each series across the whole
    ``macro_annual`` table**, used for years the series does not cover (EIBOR has
    9 of 15 years - the CBUAE rate page is Cloudflare-blocked, so it comes from a
    manual export). A zero fill was rejected: a 0.00% interbank rate is not a
    missing value, it is a false one, and trees would split on the gap as if it
    were real.

    The median is computed from the fixed 15-row macro table rather than from a
    training fold, so train and serve agree without shipping fitted state
    alongside the model. These are exogenous series, not target-derived, so this
    is not target leakage.
    """
    from db.queries.macro import all_macro_annual

    rows = all_macro_annual()
    if not rows:
        return {}, [], {}

    available = [f for f in MACRO_FEATURES if any(r.get(f) is not None for r in rows)]
    by_year: dict[int, dict] = {}
    for r in rows:
        year = r.get("year")
        if year is None:
            continue
        by_year[int(year)] = {f: r.get(f) for f in available}

    impute = {}
    for f in available:
        med = _median([r.get(f) for r in rows])
        if med is not None:
            impute[f] = med
    return by_year, available, impute


def join_macro_annual(df):
    """LEFT JOIN the macro series onto a feature-matrix DataFrame by ``year``.

    Returns ``(df, available_features)``. A LEFT join is deliberate: the sales
    record starts in 2012 and macro coverage is not guaranteed for every year, so
    an inner join would silently drop training rows to match macro availability -
    changing the comparison's row set and making the phase-1/phase-2 delta
    meaningless. Row count is asserted unchanged.
    """
    by_year, available, impute = macro_by_year()
    if not available:
        log.warning("macro_join_skipped", reason="no macro_annual rows")
        return df, []

    n_before = len(df)
    for feat in available:
        df[feat] = [
            by_year.get(int(y), {}).get(feat) if y is not None else None
            for y in df["year"]
        ]
        if feat in impute:
            df[feat] = df[feat].fillna(impute[feat])
    assert len(df) == n_before, "macro join changed the row count"

    covered = sum(1 for y in df["year"].unique() if int(y) in by_year)
    log.info(
        "macro_joined",
        features=len(available),
        years_covered=covered,
        years_total=df["year"].nunique(),
    )
    return df, available


def macro_row_for_year(year: int) -> dict:
    """Macro feature values for one year, for inference. Never raises.

    Falls back to the series medians for a year outside the ingested range, so a
    prediction for an unrepresented year degrades to "typical conditions" rather
    than to zeros.
    """
    try:
        by_year, available, impute = macro_by_year()
    except Exception as exc:  # noqa: BLE001 - inference must not die on macro
        log.warning("macro_lookup_failed", error=str(exc))
        return {}
    if not available:
        return {}
    row = dict(by_year.get(int(year), {}))
    for feat in available:
        if row.get(feat) is None and feat in impute:
            row[feat] = impute[feat]
    return row
