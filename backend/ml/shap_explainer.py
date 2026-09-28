"""SHAP explainability (M3C-01, M6-04).

Wraps shap.TreeExplainer for LightGBM. Produces global feature importance
(mean |SHAP|), per-prediction attributions, and the Phase 2 two-level
decomposition into property vs macro/seasonal groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from ml.features import MACRO_FEATURES

log = structlog.get_logger(__name__)


def _explainer(model):
    import shap

    return shap.TreeExplainer(model)


_SUMMARY_SAMPLE_ROWS = 400


def compute_summary(model, X: pd.DataFrame) -> list[dict]:
    """Mean |SHAP| per feature, ranked descending (computed on a row sample)."""
    X_sample = (
        X.sample(_SUMMARY_SAMPLE_ROWS, random_state=42)
        if len(X) > _SUMMARY_SAMPLE_ROWS
        else X
    )
    shap_values = _explainer(model).shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(X.columns, mean_abs), key=lambda t: t[1], reverse=True
    )
    return [
        {
            "feature_name": name,
            "mean_abs_shap": round(float(val), 5),
            "rank": i + 1,
            "feature_group": _group_for(name),
        }
        for i, (name, val) in enumerate(ranked)
    ]


def explain_single(model, x_row: pd.DataFrame) -> dict[str, float]:
    """SHAP values for one prediction → {feature: contribution}."""
    values = _explainer(model).shap_values(x_row)
    row = values[0] if values.ndim > 1 else values
    return {col: round(float(v), 5) for col, v in zip(x_row.columns, row)}


def decompose_two_level(shap_dict: dict[str, float]) -> dict[str, dict]:
    """Split a SHAP dict into property vs macro/seasonal groups (M6-04)."""
    prop, macro = {}, {}
    for feat, val in shap_dict.items():
        if _group_for(feat) == "macro":
            macro[feat] = val
        else:
            prop[feat] = val
    return {"property": prop, "macro": macro}


def _group_for(feature_name: str) -> str:
    """Macro features keep their group even after one-hot prefixing."""
    for mf in MACRO_FEATURES:
        if feature_name == mf or feature_name.startswith(mf):
            return "macro"
    return "property"
