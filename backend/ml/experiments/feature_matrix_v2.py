"""EXPERIMENTAL feature matrix V2 - bedroom-level grain + per-sqm features (F11).

V2 was adopted as the production builder, ``db/queries/feature_matrix_dld.py``,
so this module now delegates to it instead of carrying a second copy of the SQL.
The only difference from the experiment as run is that production also LEFT-joins
the OSM pair (``nearest_metro_km`` / ``poi_density_1km``); the V2 runner does not
use those columns, so its results are unchanged.

What V2 changed versus the original V1 builder:
  * Grain: **(area × canonical_type × bedrooms × year)** - adds ``bedrooms``.
    Bedrooms is a real column on BOTH sides (no imputation):
      - sales: parsed from ``rooms_en``  ("1 B/R"→1, "Studio"→0).
      - rents: parsed from ``ejari_property_sub_type_en``
               ("1bed room+Hall"→1, "2 bed rooms+hall"→2, "Studio"→0).
  * Features add **price_per_sqm** and **rent_per_sqm**. These share the
    target's own medians and leak it; see F45 and ``run_leakage_check``.
"""

from __future__ import annotations

from db.queries.feature_matrix_dld import build_feature_matrix


def build_feature_matrix_v2(feature_version: int = 2) -> list[dict]:
    """Return one dict per (area, type, bedrooms, year) training row."""
    return build_feature_matrix(feature_version=feature_version)
