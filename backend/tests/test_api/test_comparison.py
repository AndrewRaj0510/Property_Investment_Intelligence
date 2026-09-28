"""Phase 1 vs Phase 2 - the Aim 3 comparison.

Pins the behaviour that makes the comparison meaningful, not the specific
numbers (those move when either model is retrained). The one thing these tests
must never allow is the two phases quietly becoming the same model, which would
make the delta read 0.0000 and look like a finding.
"""

from __future__ import annotations

import pytest

from ml.macro_features import MACRO_FEATURES, macro_row_for_year
from ml.persist import active_run_row


def test_macro_feature_list_is_not_empty():
    """Phase 2 is defined by these features; an empty list silently makes it
    identical to Phase 1 while still reporting itself as 'phase 2'."""
    assert len(MACRO_FEATURES) >= 10
    assert "eibor_3m" in MACRO_FEATURES
    assert "thermal_index" in MACRO_FEATURES


def test_macro_features_are_grouped_as_macro_for_shap():
    """The two-level SHAP split must recognise every macro feature.

    If grouping drifts from the feature list, a phase-2 model would report its
    macro drivers as property drivers and the split would read 100/0 - the same
    output a phase-1 model gives, but for the wrong reason.
    """
    from ml.shap_explainer import _group_for

    for feat in MACRO_FEATURES:
        assert _group_for(feat) == "macro", feat
    for feat in ("bedrooms", "price_per_sqm", "community_Marsa Dubai"):
        assert _group_for(feat) == "property", feat


def test_encoding_includes_macro_only_when_present():
    """One encoder serves both phases via intersect-with-present.

    A phase-1 frame carries no macro columns, so they must not appear in its
    encoded output - otherwise the baseline silently gains phase-2 features.
    """
    import pandas as pd

    from ml.encoding import encode

    base = {"year": 2024, "bedrooms": 2, "price_per_sqm": 100.0}
    x1 = encode(pd.DataFrame([base]))
    assert not any(m in x1.columns for m in MACRO_FEATURES)

    x2 = encode(pd.DataFrame([{**base, "eibor_3m": 4.2, "thermal_index": 39.0}]))
    assert "eibor_3m" in x2.columns
    assert "thermal_index" in x2.columns


def test_macro_row_for_year_never_returns_zeros_for_missing_series():
    """Uncovered years fall back to the series median, not 0.

    A 0.00% interbank rate is a false value, not a missing one, and a tree will
    split on it as though it were real.
    """
    row = macro_row_for_year(1900)
    if row:
        assert row.get("eibor_3m", 1) != 0


def test_comparison_endpoint_shape(client):
    r = client.get("/api/v1/models/comparison")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    if not body["available"]:
        assert body["detail"]
        return

    p1, p2, delta = body["phase1"], body["phase2"], body["delta"]
    assert p1["run_name"].startswith("phase1")
    assert p2["run_name"].startswith("phase2")

    assert p2["n_features"] > p1["n_features"]
    assert body["macro_feature_count"] >= 10

    assert delta["r_squared"] == pytest.approx(
        p2["r_squared"] - p1["r_squared"], abs=1e-4
    )
    assert body["interpretation"]


def test_comparison_reports_attribution_and_accuracy_separately(client):
    """The two headline numbers answer different questions and must both exist.

    Reporting attribution share alone would imply macro matters; reporting the
    accuracy delta alone would imply it is absent.
    """
    body = client.get("/api/v1/models/comparison").json()
    if not body["available"]:
        pytest.skip("no phase-2 model trained in this environment")
    assert body["macro_attribution_pct"] is not None
    assert body["delta"]["r_squared"] is not None
    assert len(body["macro_drivers"]) >= 1


@pytest.mark.skipif(
    active_run_row(phase=2) is None, reason="no phase-2 model trained"
)
def test_predict_phase2_differs_from_phase1(client):
    """The extended model must produce its own prediction and a real SHAP split.

    A phase-1 model decomposes to 100% property because it has no macro
    features. A phase-2 model has them, so its macro group must be non-empty -
    that difference is the measured Objective 3.5 result.
    """
    body = {
        "community": "Dubai Marina",
        "property_type": "apartment",
        "bedrooms": 2,
        "floor_area_sqft": 1200,
    }
    r1 = client.post("/api/v1/predict", json={**body, "use_extended_model": False})
    r2 = client.post("/api/v1/predict", json={**body, "use_extended_model": True})
    assert r1.status_code == 200 and r2.status_code == 200
    j1, j2 = r1.json(), r2.json()

    assert j1["model_phase"] == 1 and j2["model_phase"] == 2
    assert j1["model_run_id"] != j2["model_run_id"]

    assert j1["shap_explanation"]["macro_level"] == []
    assert j1["shap_explanation"]["property_share_pct"] == 100.0

    assert len(j2["shap_explanation"]["macro_level"]) >= 1
    assert j2["shap_explanation"]["property_share_pct"] < 100.0
