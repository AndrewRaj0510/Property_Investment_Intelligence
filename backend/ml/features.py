"""Feature matrix adapter (Phase 1, real DLD data).

The Phase-1 feature matrix is one row per **(community/area, canonical property
type, bedrooms, year)**, built directly from the raw DLD sales + Ejari rent tables by
``db/queries/feature_matrix_dld.py`` (which shares its type-normalization and
yield-clamp logic with the live-scoring baseline in ``db/queries/yield_agg.py``).
The model's feature list lives in ``ml/encoding.py``, shared by training and
inference.

This module is the thin adapter the pipeline uses:
    * ``to_records``     → the subset of columns persisted to the
                           ``feature_matrix`` table.
    * ``MACRO_FEATURES`` → re-exported for the SHAP property/macro grouping.

The macro join happens in ``ml.train.train_and_evaluate``, not in the matrix: the
persisted ``feature_matrix`` table is rebuilt on every training run, so writing
phase-specific columns into it would mean a phase-1 run silently stripped the
columns a phase-2 model needs at inference. Keeping the table phase-neutral lets
both models coexist.
"""

from __future__ import annotations

from ml.macro_features import MACRO_FEATURES
from ml.persist import FEATURE_MATRIX_COLUMNS

__all__ = ["MACRO_FEATURES", "to_records"]


def to_records(rows: list[dict]) -> list[dict]:
    """Project rows to the columns stored in the ``feature_matrix`` table."""
    return [{c: r.get(c) for c in FEATURE_MATRIX_COLUMNS} for r in rows]
