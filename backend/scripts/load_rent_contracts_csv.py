"""Load a real DLD Ejari rent-contracts CSV (from the DDA portal) into Postgres.

The DDA staging API masks the rent dataset (scrambled area names, fake values),
so for the proof-of-concept we ingest the **real** rent-contracts export
downloaded from the DDA open-data portal instead. One file is ~1M rows / ~500 MB,
so this streams the CSV in chunks (never loads it whole) and bulk-loads via
Postgres ``COPY`` - far faster than row-by-row inserts.

It targets the SAME table the API path uses (``dld_rent_contracts_raw``): the
CSV's 41 columns are exactly the 41 source columns already there, so no schema
change is needed. Because the previously-loaded staging rows are fake and can't
join to real sales, this **replaces** the table contents by default.

For an investment-yield PoC we keep **Residential** usage only (drops Commercial
/ Labor Camps / Office / Industrial noise), which is ~70% of the file and all the
relevant rows for scoring residential property.

Run from the backend/ directory:
    .venv/Scripts/python.exe -m scripts.load_rent_contracts_csv data/<file>.csv

Options (env or args):
    --all-usage       load every property_usage (default: Residential only)
    --append          keep existing rows (default: replace/TRUNCATE first)
    --limit N         stop after N kept rows (for a quick test load)
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

TABLE = "dld_rent_contracts_raw"
LOADED_AT_COL = "_loaded_at"
KEY_COL = "contract_id"
USAGE_COL = "property_usage_en"
RESIDENTIAL = "Residential"
TYPE_COL = "ejari_property_type_en"
SUBTYPE_COL = "ejari_property_sub_type_en"
AREA_COL = "area_name_en"
DATE_COL = "contract_start_date"
MIN_YEAR = 2012
MAX_YEAR = 2026
KEEP_TYPES = {
    "Flat",
    "Studio",
    "Villa",
    "Complex Villas",
    "Penthouse",
    "Hotel apartments",
    "Arabian House",
    "Building",
}
PER_BUCKET_CAP = 100
BATCH = 20_000
csv.field_size_limit(10_000_000)


def _table_exists(cur) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f"public.{TABLE}",))
    return cur.fetchone()[0] is not None


def _create_table_sql(columns: list[str]) -> str:
    col_defs = ",\n    ".join(f"{_quote_ident(c)} TEXT" for c in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {_quote_ident(TABLE)} (\n"
        f"    {col_defs},\n"
        f"    {_quote_ident(LOADED_AT_COL)} TIMESTAMPTZ DEFAULT NOW()\n"
        f");"
    )


def _drop_unique_constraint_sql() -> str:
    """Drop the API-era UNIQUE(contract_id) if present.

    In the REAL Ejari data ``contract_id`` is NOT unique - a contract can have
    multiple rows (multi-property contracts via ``line_number``, renewals,
    amendments); this file has ~108k repeats over ~1M rows. The API path added a
    UNIQUE(contract_id) for its upsert dedup, but that constraint is wrong for the
    real dataset and would make ``COPY`` fail. We compute medians over all rent
    rows, so uniqueness isn't needed; drop it so COPY can bulk-load.
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
    residential_only: bool = True,
    replace: bool = True,
    limit: int | None = None,
    keep_types: set[str] | None = KEEP_TYPES,
    per_bucket_cap: int | None = PER_BUCKET_CAP,
) -> None:
    """Stream-load the rent CSV into Postgres, filtered + sampled to fit quota.

    ``keep_types`` (None = all) keeps only investment-relevant property types.
    ``per_bucket_cap`` (None = no cap) limits rows per (area, type, sub_type) so
    the table stays small enough for a tight Supabase quota while every bucket
    keeps a representative, median-stable sample.
    """
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
    usage_idx = header.index(USAGE_COL) if USAGE_COL in header else None
    type_idx = header.index(TYPE_COL) if TYPE_COL in header else None
    area_idx = header.index(AREA_COL) if AREA_COL in header else None
    date_idx = header.index(DATE_COL) if DATE_COL in header else None
    kept = skipped_usage = skipped_type = skipped_cap = skipped_year = 0
    bucket_counts: dict[tuple[str, str, str], int] = {}

    def _year_of(row: list[str]) -> str:
        """Start-year as a 4-char string, or '' if unparseable."""
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
                if usage_idx is not None and residential_only:
                    if row[usage_idx] != RESIDENTIAL:
                        skipped_usage += 1
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
                    print(f"  loaded {kept:,} (usage {skipped_usage:,}, "
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
    log.info("csv_load_done", kept=kept, skipped_usage=skipped_usage,
             skipped_type=skipped_type, skipped_year=skipped_year,
             skipped_cap=skipped_cap, table_rows=total, buckets=len(bucket_counts))
    print(f"Done. Kept {kept:,} across {len(bucket_counts):,} (area,type,year) "
          f"buckets. Skipped: {skipped_usage:,} non-residential, "
          f"{skipped_type:,} non-investment type, {skipped_year:,} out-of-year-range, "
          f"{skipped_cap:,} over per-bucket cap. "
          f"Table now {total:,} rows ({distinct:,} distinct {KEY_COL}; "
          f"{total - distinct:,} repeats - expected in real Ejari data).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a DLD rent-contracts CSV.")
    ap.add_argument("csv_path", help="Path to the rent-contracts CSV.")
    ap.add_argument("--all-usage", action="store_true",
                    help="Load all property_usage rows (default: Residential only).")
    ap.add_argument("--append", action="store_true",
                    help="Keep existing rows (default: replace/TRUNCATE first).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N kept rows (quick test).")
    args = ap.parse_args()
    load_csv(
        args.csv_path,
        residential_only=not args.all_usage,
        replace=not args.append,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
