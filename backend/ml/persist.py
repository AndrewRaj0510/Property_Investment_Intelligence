"""psycopg-based persistence for the training pipeline.

The Supabase ORM client (``SUPABASE_URL`` / ``SUPABASE_KEY``) is not configured
in this project - every DB interaction goes through the direct Postgres
connection (the ``SUPABASE_DB_*`` parts via the Session Pooler), the same
connection the ingestion scripts use. This module writes the three ML tables
(``feature_matrix``, ``model_runs``, ``shap_summary``) over that connection so
the training run is fully persisted without needing the ORM layer.

Tables are created idempotently here (they were previously only defined in
``sql/schema.sql`` for the ORM path, which was never applied against this DB).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog

from config import get_settings
from db.pg import connect as _connect

log = structlog.get_logger(__name__)


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS feature_matrix (
        id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        community              TEXT NOT NULL,
        property_type          TEXT NOT NULL,
        bedrooms               INTEGER NOT NULL,
        year                   INTEGER NOT NULL,
        gross_yield_lt         NUMERIC,
        sale_price_median_aed  NUMERIC,
        annual_rent_median_aed NUMERIC,
        floor_area_median_sqm  NUMERIC,
        price_per_sqm          NUMERIC,
        rent_per_sqm           NUMERIC,
        nearest_metro          TEXT,
        parking_rate           NUMERIC,
        n_sales                INTEGER,
        n_rents                INTEGER,
        nearest_metro_km       NUMERIC,
        poi_density_1km        INTEGER,
        feature_version        INTEGER DEFAULT 1,
        computed_at            TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (community, property_type, bedrooms, year, feature_version)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_name        TEXT NOT NULL,
        phase           INTEGER NOT NULL,
        model_type      TEXT NOT NULL,
        feature_version INTEGER,
        mae             NUMERIC,
        rmse            NUMERIC,
        r_squared       NUMERIC,
        cv_folds        INTEGER DEFAULT 5,
        artefact_path   TEXT,
        params_json     JSONB,
        is_active       BOOLEAN DEFAULT FALSE,
        trained_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS shap_summary (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        model_run_id    UUID REFERENCES model_runs(id),
        feature_name    TEXT NOT NULL,
        mean_abs_shap   NUMERIC,
        rank            INTEGER,
        feature_group   TEXT
    );
    """,
]

FEATURE_MATRIX_COLUMNS = [
    "community",
    "property_type",
    "bedrooms",
    "year",
    "gross_yield_lt",
    "sale_price_median_aed",
    "annual_rent_median_aed",
    "floor_area_median_sqm",
    "price_per_sqm",
    "rent_per_sqm",
    "nearest_metro",
    "parking_rate",
    "n_sales",
    "n_rents",
    "nearest_metro_km",
    "poi_density_1km",
    "feature_version",
]


def ensure_ml_tables() -> None:
    """Create feature_matrix / model_runs / shap_summary if absent (idempotent).

    ``feature_matrix`` is a *derived* table (rebuilt from the raw DLD tables every
    run), so when its schema evolves we drop and recreate it - ``CREATE TABLE IF
    NOT EXISTS`` alone cannot add the new bedroom/per-sqm columns to a pre-existing
    table. This is safe: no source data lives here. model_runs/shap_summary are
    never dropped (they hold run history).
    """
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feature_matrix")
            for ddl in _DDL:
                cur.execute(ddl)
        conn.commit()
    log.info("ml_tables_ensured")


def replace_feature_matrix_rows(rows: list[dict], feature_version: int) -> int:
    """Delete this version's rows then bulk-insert the freshly built matrix."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM feature_matrix WHERE feature_version = %s",
                (feature_version,),
            )
            if rows:
                placeholders = "(" + ", ".join(["%s"] * len(FEATURE_MATRIX_COLUMNS)) + ")"
                cols_sql = ", ".join(FEATURE_MATRIX_COLUMNS)
                values = [tuple(r.get(c) for c in FEATURE_MATRIX_COLUMNS) for r in rows]
                cur.executemany(
                    f"INSERT INTO feature_matrix ({cols_sql}) VALUES {placeholders}",
                    values,
                )
        conn.commit()
    log.info("feature_matrix_persisted", rows=len(rows), version=feature_version)
    return len(rows)


def insert_model_run(row: dict) -> str:
    """Insert a model_runs row; return its generated id (uuid str)."""
    run_id = str(uuid.uuid4())
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs
                    (id, run_name, phase, model_type, feature_version,
                     mae, rmse, r_squared, cv_folds, artefact_path,
                     params_json, is_active, trained_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    row["run_name"],
                    row["phase"],
                    row["model_type"],
                    row.get("feature_version"),
                    row.get("mae"),
                    row.get("rmse"),
                    row.get("r_squared"),
                    row.get("cv_folds", 5),
                    row.get("artefact_path"),
                    json.dumps(row.get("params_json", {})),
                    row.get("is_active", False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    log.info("model_run_inserted", run_id=run_id, run_name=row["run_name"])
    return run_id


def replace_shap_summary(run_id: str, summary: list[dict]) -> int:
    """Replace the SHAP summary rows for a run."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM shap_summary WHERE model_run_id = %s", (run_id,))
            values = [
                (run_id, s["feature_name"], s["mean_abs_shap"], s["rank"],
                 s.get("feature_group"))
                for s in summary
            ]
            cur.executemany(
                """
                INSERT INTO shap_summary
                    (model_run_id, feature_name, mean_abs_shap, rank, feature_group)
                VALUES (%s,%s,%s,%s,%s)
                """,
                values,
            )
        conn.commit()
    log.info("shap_summary_persisted", run_id=run_id, features=len(summary))
    return len(summary)


def activate_run(run_id: str, phase: int) -> None:
    """Deactivate every other run for this phase, then activate this one."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE model_runs SET is_active = FALSE WHERE phase = %s", (phase,)
            )
            cur.execute(
                "UPDATE model_runs SET is_active = TRUE WHERE id = %s", (run_id,)
            )
        conn.commit()
    log.info("model_run_activated", run_id=run_id, phase=phase)


_RUN_COLS = [
    "id",
    "run_name",
    "phase",
    "model_type",
    "feature_version",
    "mae",
    "rmse",
    "r_squared",
    "cv_folds",
    "artefact_path",
    "is_active",
]


def active_run_row(phase: int | None = None) -> dict | None:
    """Return the active model_runs row (optionally for a phase), or None."""
    settings = get_settings()
    cols = ", ".join(_RUN_COLS)
    sql = f"SELECT {cols} FROM model_runs WHERE is_active = TRUE"
    params: list = []
    if phase is not None:
        sql += " AND phase = %s"
        params.append(phase)
    sql += " ORDER BY trained_at DESC LIMIT 1"
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    if not row:
        return None
    out = dict(zip(_RUN_COLS, row))
    out["id"] = str(out["id"])
    return out


def feature_matrix_rows(feature_version: int | None = None) -> list[dict]:
    """Return feature_matrix rows (optionally for a version) as dicts."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            if feature_version is not None:
                cur.execute(
                    f"SELECT {', '.join(FEATURE_MATRIX_COLUMNS)} FROM feature_matrix "
                    "WHERE feature_version = %s",
                    (feature_version,),
                )
            else:
                cur.execute(f"SELECT {', '.join(FEATURE_MATRIX_COLUMNS)} FROM feature_matrix")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, rec)) for rec in cur.fetchall()]


def shap_summary_rows(model_run_id: str) -> list[dict]:
    """Return the stored SHAP summary for a run, ranked, as dicts (psycopg read).

    The ORM reader (db.queries.shap_values.summary_for_run) is unusable here since
    the ORM isn't configured; the summary is written via replace_shap_summary, so
    this reads it back over the same psycopg connection.
    """
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT feature_name, mean_abs_shap, rank, feature_group "
                "FROM shap_summary WHERE model_run_id = %s ORDER BY rank",
                (model_run_id,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, rec)) for rec in cur.fetchall()]
    for r in rows:
        if r.get("mean_abs_shap") is not None:
            r["mean_abs_shap"] = float(r["mean_abs_shap"])
    return rows
