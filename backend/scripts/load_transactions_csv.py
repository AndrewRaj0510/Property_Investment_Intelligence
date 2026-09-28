"""Load a real DLD Transactions (sales) CSV from the DDA portal into Postgres.

The sales counterpart of ``load_rent_contracts_csv.py``. The per-date staging
API only reached ~2 months of 2025; the DDA portal's downloadable Transactions
CSV carries the full history (1989-2026, ~1.7M rows across files), which is what
the time-series / yield model needs. This streams the CSV(s) in chunks and
bulk-loads via ``COPY`` into the SAME table the API path used
(``dld_transactions_raw``) - the CSV's 47 columns match its source columns, so
no schema change.

Balancing (so the model sees ~equal samples per investment type, not 6:1
apartments): keep only real **Sales** (drop Mortgages/Gifts), keep investment
property types (drop Land), restrict to 2012-2026 (matches the rent history),
and cap each **(area, property_type, year)** bucket at ``PER_BUCKET_CAP``.

Run from backend/ (replace mode by default; --append to accumulate files):
    .venv/Scripts/python.exe -m scripts.load_transactions_csv data/<file>.csv
    .venv/Scripts/python.exe -m scripts.load_transactions_csv data/<file>.csv --append
"""

from __future__ import annotations

import argparse
import csv
import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)

TABLE = "dld_transactions_raw"
LOADED_AT_COL = "_loaded_at"
KEY_COL = "transaction_id"
TYPE_COL = "property_type_en"
AREA_COL = "area_name_en"
DATE_COL = "instance_date"
GROUP_COL = "trans_group_en"
KEEP_TYPES = {"Unit", "Villa", "Building"}
KEEP_GROUP = "Sales"
MIN_YEAR = 2012
MAX_YEAR = 2026
PER_BUCKET_CAP = 100
BATCH = 20_000
csv.field_size_limit(10_000_000)


def _create_table_sql(columns: list[str]) -> str:
    col_defs = ",\n    ".join(f"{_quote_ident(c)} TEXT" for c in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {_quote_ident(TABLE)} (\n"
        f"    {col_defs},\n"
        f"    {_quote_ident(LOADED_AT_COL)} TIMESTAMPTZ DEFAULT NOW()\n"
        f");"
    )


def _drop_unique_constraint_sql() -> str:
    """Drop the API-era UNIQUE(transaction_id).

    The API path added it for upsert dedup, but the bulk CSV can legitimately
    repeat a transaction_id across procedure rows, and COPY has no ON CONFLICT.
    Medians don't need uniqueness, so drop it to allow fast COPY loads.
    """
    return (
        f"ALTER TABLE {_quote_ident(TABLE)} "
        f"DROP CONSTRAINT IF EXISTS {_quote_ident(TABLE + '_' + KEY_COL + '_key')};"
    )


def _copy_sql(columns: list[str]) -> str:
    cols_sql = ", ".join(_quote_ident(c) for c in columns)
    return f"COPY {_quote_ident(TABLE)} ({cols_sql}) FROM STDIN"


def load_csv(
    path: str,
    replace: bool = True,
    limit: int | None = None,
    keep_types: set[str] | None = KEEP_TYPES,
    per_bucket_cap: int | None = PER_BUCKET_CAP,
) -> None:
    settings = get_settings()
    if not (settings.db_parts_configured or settings.supabase_db_url):
        raise SystemExit("No Postgres connection configured in .env.")

    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    log.info("csv_header_read", columns=len(header))

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_create_table_sql(header))
            cur.execute(_drop_unique_constraint_sql())
            if replace:
                cur.execute(f"TRUNCATE {_quote_ident(TABLE)};")
                print(f"Cleared {TABLE} (replace mode).")
        conn.commit()

    copy_stmt = _copy_sql(header)
    type_idx = header.index(TYPE_COL) if TYPE_COL in header else None
    area_idx = header.index(AREA_COL) if AREA_COL in header else None
    date_idx = header.index(DATE_COL) if DATE_COL in header else None
    group_idx = header.index(GROUP_COL) if GROUP_COL in header else None
    kept = skipped_group = skipped_type = skipped_year = skipped_cap = 0
    bucket_counts: dict[tuple[str, str, str], int] = {}

    def _year_of(row: list[str]) -> str:
        if date_idx is None:
            return ""
        v = row[date_idx]
        return v[:4] if len(v) >= 4 and v[:4].isdigit() else ""

    if not replace and per_bucket_cap is not None:
        with _connect(settings) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_quote_ident(AREA_COL)}, {_quote_ident(TYPE_COL)}, "
                f"left({_quote_ident(DATE_COL)}, 4), count(*) "
                f"FROM {_quote_ident(TABLE)} GROUP BY 1, 2, 3;"
            )
            for a, t, y, n in cur.fetchall():
                bucket_counts[(a or "", t or "", y or "")] = n
        log.info("append_mode_seeded_buckets", existing_buckets=len(bucket_counts))

    with _connect(settings) as conn:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            batch: list[list[str]] = []

            def flush(rows: list[list[str]]) -> None:
                if not rows:
                    return
                with conn.cursor() as cur:
                    with cur.copy(copy_stmt) as cp:
                        for r in rows:
                            cp.write_row(r)
                conn.commit()

            for row in reader:
                if group_idx is not None and row[group_idx] != KEEP_GROUP:
                    skipped_group += 1
                    continue
                if keep_types is not None and type_idx is not None:
                    if row[type_idx] not in keep_types:
                        skipped_type += 1
                        continue
                year = _year_of(row)
                if date_idx is not None:
                    if not year or not (MIN_YEAR <= int(year) <= MAX_YEAR):
                        skipped_year += 1
                        continue
                if per_bucket_cap is not None and area_idx is not None:
                    key = (
                        row[area_idx],
                        row[type_idx] if type_idx is not None else "",
                        year,
                    )
                    if bucket_counts.get(key, 0) >= per_bucket_cap:
                        skipped_cap += 1
                        continue
                    bucket_counts[key] = bucket_counts.get(key, 0) + 1
                batch.append(row)
                kept += 1
                if len(batch) >= BATCH:
                    flush(batch)
                    batch = []
                    print(f"  loaded {kept:,} (group {skipped_group:,}, "
                          f"type {skipped_type:,}, year {skipped_year:,}, "
                          f"cap {skipped_cap:,})", end="\r")
                if limit is not None and kept >= limit:
                    break
            flush(batch)

    print()
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*), count(distinct {_quote_ident(KEY_COL)}) "
                    f"FROM {_quote_ident(TABLE)};")
        total, distinct = cur.fetchone()
    log.info("csv_load_done", kept=kept, skipped_group=skipped_group,
             skipped_type=skipped_type, skipped_year=skipped_year,
             skipped_cap=skipped_cap, table_rows=total, buckets=len(bucket_counts))
    print(f"Done. Kept {kept:,} across {len(bucket_counts):,} (area,type,year) "
          f"buckets. Skipped: {skipped_group:,} non-Sales, "
          f"{skipped_type:,} non-investment type, {skipped_year:,} out-of-year-range, "
          f"{skipped_cap:,} over per-bucket cap. "
          f"Table now {total:,} rows ({distinct:,} distinct {KEY_COL}).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a DLD Transactions (sales) CSV.")
    ap.add_argument("csv_path", help="Path to the transactions CSV.")
    ap.add_argument("--append", action="store_true",
                    help="Keep existing rows (default: replace/TRUNCATE first).")
    ap.add_argument("--all-types", action="store_true",
                    help="Load all property types (default: Unit/Villa/Building).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N kept rows (quick test).")
    args = ap.parse_args()
    load_csv(
        args.csv_path,
        replace=not args.append,
        limit=args.limit,
        keep_types=None if args.all_types else KEEP_TYPES,
    )


if __name__ == "__main__":
    main()
