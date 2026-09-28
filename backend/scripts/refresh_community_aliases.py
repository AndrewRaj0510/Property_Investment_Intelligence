"""Materialize marketing-name → DLD-area aliases into ``community_aliases`` (F21).

Mines ``master_project_en`` → ``area_name_en`` from the raw DLD transactions, so
the mapping is evidence from the data rather than hand-asserted geography (e.g.
*Jumeirah Village Circle → Al Barsha South Fourth*, backed by 3,305 rows).

Run offline/periodically, alongside the yield-median refresh, from backend/:
    .venv/Scripts/python.exe -m scripts.refresh_community_aliases

Re-run it after ingesting more DLD data - more transactions mean more master
projects clear the sample threshold and more portal names become resolvable.
"""

from __future__ import annotations

import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from db.queries.community_aliases import ALIAS_TABLE, compute_aliases
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)


def _create_table_sql() -> str:
    t = _quote_ident(ALIAS_TABLE)
    return f"""
    CREATE TABLE IF NOT EXISTS {t} (
        alias           TEXT PRIMARY KEY,
        area            TEXT NOT NULL,
        n_transactions  INTEGER NOT NULL,
        -- Transactions for this alias that fell in OTHER areas. 0 = the
        -- marketing name maps cleanly to one DLD area; >0 means it spans
        -- several and we took the dominant one, so the ambiguity stays visible.
        n_other_areas   INTEGER NOT NULL DEFAULT 0,
        refreshed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """


def main() -> int:
    rows = compute_aliases()
    if not rows:
        print("No aliases derived - is dld_transactions_raw populated?")
        return 1

    t = _quote_ident(ALIAS_TABLE)
    with _connect(get_settings()) as conn, conn.cursor() as cur:
        cur.execute(_create_table_sql())
        cur.execute(f"TRUNCATE {t}")
        cur.executemany(
            f"INSERT INTO {t} (alias, area, n_transactions, n_other_areas) "
            f"VALUES (%(alias)s, %(area)s, %(n_transactions)s, %(n_other_areas)s)",
            rows,
        )
        conn.commit()

    ambiguous = [r for r in rows if r["n_other_areas"] > 0]
    print(f"Wrote {len(rows)} aliases to {ALIAS_TABLE}.")
    print(f"  unambiguous : {len(rows) - len(ambiguous)}")
    print(f"  spanning >1 area (dominant kept): {len(ambiguous)}")
    print("\nTop 10 by transaction count:")
    for r in rows[:10]:
        print(f"  {r['alias']:<38} -> {r['area']:<28} ({r['n_transactions']})")
    if ambiguous:
        print("\nMost ambiguous (check these if a verdict looks wrong):")
        for r in sorted(ambiguous, key=lambda x: -x["n_other_areas"])[:5]:
            print(
                f"  {r['alias']:<38} -> {r['area']:<28} "
                f"({r['n_transactions']} here, {r['n_other_areas']} elsewhere)"
            )
    log.info("community_aliases_refreshed", n=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
