"""Shared feature encoding for training and inference.

Keeping encoding in one place guarantees the column set the model was trained
on matches the column set produced at inference time.

Feature set (Phase 1) is derived from the real DLD feature matrix
(``db/queries/feature_matrix_dld.py``): one row per (community/area, canonical
property type, bedrooms, year). Location signal comes from ``community`` and
``nearest_metro`` (both one-hot) plus the OpenStreetMap pair ``nearest_metro_km``
/ ``poi_density_1km``; the property is described by bedrooms, size, price/rent
per sqm (density), parking rate, and the transaction year; sample sizes give the
model a notion of how well-evidenced a bucket is.

Adding the bedroom grain + per-sqm features raised 5-fold-CV R² from 0.654 to
0.825 (see ``artifacts/DISSERTATION_FINDINGS.md`` F11).
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL = ["property_type", "community", "nearest_metro"]

NUMERIC = [
    "year",
    "bedrooms",
    "floor_area_median_sqm",
    "price_per_sqm",
    "rent_per_sqm",
    "parking_rate",
    "n_sales",
    "n_rents",
    "nearest_metro_km",
    "poi_density_1km",
]

from ml.macro_features import MACRO_FEATURES  # noqa: E402

NUMERIC = NUMERIC + MACRO_FEATURES

_IMPUTE = {
    "bedrooms": 1.0,
    "floor_area_median_sqm": 0.0,
    "price_per_sqm": 0.0,
    "rent_per_sqm": 0.0,
    "parking_rate": 0.0,
    "nearest_metro_km": 99.0,
    "poi_density_1km": 0.0,
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, keep present numerics. Returns feature frame."""
    present_num = [c for c in NUMERIC if c in df.columns]
    present_cat = [c for c in CATEGORICAL if c in df.columns]

    X = df[present_num].copy()
    for c in present_num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(_IMPUTE.get(c, 0))

    if present_cat:
        dummies = pd.get_dummies(df[present_cat].astype("object"), prefix=present_cat)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X


def align_columns(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reindex X to the model's training columns, filling missing with 0."""
    return X.reindex(columns=feature_names, fill_value=0)
