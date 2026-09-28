"""GET /api/v1/models/comparison - Phase 1 vs Phase 2 (Aim 3).

The project's central comparison: a baseline model built from property and
locational features, against the same baseline **plus** the external
macroeconomic, environmental and seasonal series ingested by Objectives 3.1-3.3.

Both models are trained on identical rows with identical folds (seed 42), each
tuned at its own optimum, so the difference measures the FEATURES and nothing
else. Both are served simultaneously - ``activate_run`` is phase-scoped, so the
extended model never displaces the baseline that backs ``/score``.

The endpoint deliberately reports two numbers that point in different
directions, because that contrast is the finding:

  * **accuracy delta** - does adding external factors predict better?
  * **attribution share** - how much of the extended model's explanation do they
    carry?

A reader given only the second would conclude macro matters; given only the
first, that it is absent. Both together say what is actually true.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from api.rate_limit import limiter
from api.schemas.comparison import (
    MacroDriver,
    MetricDelta,
    ModelComparison,
    PhaseMetrics,
)
from ml.macro_features import MACRO_FEATURES
from ml.persist import active_run_row, shap_summary_rows

log = structlog.get_logger(__name__)
router = APIRouter()


def _is_macro(feature_name: str) -> bool:
    """Macro features keep their group after one-hot prefixing.

    None of them are categorical today, but this mirrors
    ``ml.shap_explainer._group_for`` so the two groupings cannot drift apart.
    """
    return any(feature_name == m or feature_name.startswith(m) for m in MACRO_FEATURES)


def _metrics(run: dict, n_features: int) -> PhaseMetrics:
    def _f(key):
        v = run.get(key)
        return float(v) if v is not None else None

    return PhaseMetrics(
        run_id=str(run["id"]),
        run_name=str(run.get("run_name") or ""),
        model_type=str(run.get("model_type") or ""),
        mae=_f("mae"),
        rmse=_f("rmse"),
        r_squared=_f("r_squared"),
        cv_folds=int(run["cv_folds"]) if run.get("cv_folds") is not None else None,
        n_features=n_features,
    )


def _delta(a: float | None, b: float | None) -> float | None:
    return round(a - b, 4) if a is not None and b is not None else None


def _interpret(delta: MetricDelta, macro_pct: float | None) -> str:
    """State what the two numbers mean together, without overclaiming either."""
    dr2 = delta.r_squared
    if dr2 is None or macro_pct is None:
        return "Insufficient metrics to compare the two models."

    direction = (
        "improved" if dr2 > 0.005 else "worsened" if dr2 < -0.005 else "did not change"
    )
    return (
        "Adding the external macroeconomic, environmental and seasonal series "
        "{d} cross-validated accuracy (change in R-squared {r:+.4f}), while those "
        "features carry {p:.1f}% of the extended model's total attribution. "
        "Those two results are consistent, not contradictory: the feature matrix "
        "is built at annual grain and `year` is already a model feature, so every "
        "external series is a function of something the baseline already has. The "
        "trees still split on them, which is what the attribution share reflects, "
        "but that signal is collinear with `year` rather than independent, so it "
        "buys no predictive power. Attribution share and predictive contribution "
        "are different questions."
    ).format(d=direction, r=dr2, p=macro_pct)


@router.get(
    "/models/comparison", response_model=ModelComparison, tags=["explainability"]
)
@limiter.limit("60/minute")
async def model_comparison(request: Request) -> ModelComparison:
    try:
        run1 = active_run_row(phase=1)
        run2 = active_run_row(phase=2)
    except Exception as exc:  # noqa: BLE001
        log.warning("comparison_unavailable", error=str(exc))
        return ModelComparison(available=False, detail="Model registry unavailable.")

    if run1 is None:
        return ModelComparison(
            available=False, detail="No active Phase-1 baseline model."
        )

    rows1 = shap_summary_rows(run1["id"]) or []
    p1 = _metrics(run1, len(rows1))

    if run2 is None:
        return ModelComparison(
            phase1=p1,
            available=False,
            detail=(
                "No active Phase-2 model. Train one with: "
                "python -m scripts.train_model --phase 2"
            ),
        )

    rows2 = shap_summary_rows(run2["id"]) or []
    p2 = _metrics(run2, len(rows2))

    total = sum(float(r["mean_abs_shap"]) for r in rows2)
    macro_rows = [r for r in rows2 if _is_macro(str(r["feature_name"]))]
    macro_abs = sum(float(r["mean_abs_shap"]) for r in macro_rows)
    macro_pct = round(100.0 * macro_abs / total, 2) if total else None

    drivers = [
        MacroDriver(
            feature=str(r["feature_name"]), mean_abs_shap=float(r["mean_abs_shap"])
        )
        for r in sorted(
            macro_rows, key=lambda r: float(r["mean_abs_shap"]), reverse=True
        )
    ]

    delta = MetricDelta(
        mae=_delta(p2.mae, p1.mae),
        rmse=_delta(p2.rmse, p1.rmse),
        r_squared=_delta(p2.r_squared, p1.r_squared),
    )

    return ModelComparison(
        phase1=p1,
        phase2=p2,
        delta=delta,
        macro_attribution_pct=macro_pct,
        macro_drivers=drivers,
        macro_feature_count=len(macro_rows),
        interpretation=_interpret(delta, macro_pct),
    )
