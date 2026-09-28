"""Feature matrix adapter (Phase 1, real DLD data).

The Phase-1 feature matrix is one row per **(community/area, canonical property
type, year)**, built directly from the raw DLD sales + Ejari rent tables by
``db/queries/feature_matrix_dld.py`` (which shares its type-normalization and
yield-clamp logic with the live-scoring baseline in ``db/queries/yield_agg.py``).

This module is the thin adapter the pipeline uses:
    * ``build_feature_matrix`` → the list of feature-row dicts.
    * ``to_records``           → the subset of columns persisted to the
                                 ``feature_matrix`` table.
    * ``MACRO_FEATURES``       → kept (empty in Phase 1) so the SHAP grouping
                                 logic imports cleanly; Phase 2 is frozen.

The previous mock-era builder (community/property/bedrooms joined against
scraped listings + Airbnb + OSM tables) was removed with the live-only
migration - those tables are no longer populated.
"""

from __future__ import annotations

import structlog

from db.queries.feature_matrix_dld import build_feature_matrix as _build
from ml.persist import FEATURE_MATRIX_COLUMNS

log = structlog.get_logger(__name__)

PROPERTY_FEATURES = [
    "year",
    "bedrooms",
    "floor_area_median_sqm",
    "price_per_sqm",
    "rent_per_sqm",
    "parking_rate",
    "n_sales",
    "n_rents",
    "property_type",
    "community",
    "nearest_metro",
    "nearest_metro_km",
    "poi_density_1km",
]

from ml.macro_features import MACRO_FEATURES

__all__ = [
    "MACRO_FEATURES",
    "PROPERTY_FEATURES",
    "RECORD_COLUMNS",
    "build_feature_matrix",
    "to_records",
]

RECORD_COLUMNS = FEATURE_MATRIX_COLUMNS


def build_feature_matrix(phase: int = 1, feature_version: int = 1) -> list[dict]:
    """Return the feature rows (list of dicts).

    Identical for both phases. The macro join happens later, in
    ``ml.train.train_and_evaluate``, rather than here - the persisted
    ``feature_matrix`` table is rebuilt on every training run, so writing
    phase-specific columns into it would mean a phase-1 run silently stripped the
    columns a phase-2 model needs at inference. Keeping the table phase-neutral
    lets both models coexist.
    """
    return _build(feature_version=feature_version)


def to_records(rows: list[dict]) -> list[dict]:
    """Project rows to the columns stored in the ``feature_matrix`` table."""
    out = []
    for r in rows:
        out.append({c: r.get(c) for c in RECORD_COLUMNS})
    return out
