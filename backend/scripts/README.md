# Scripts

Operational scripts for the backend. Run them from the **`backend/` directory**
(so `.env` and imports resolve).

---

## `pull_and_store_dld.py` - pull live DLD records into Supabase

Pulls real Dubai Land Department (DLD) transactions from the data.dubai (DDA)
API and stores them in the Postgres table `dld_transactions_raw`. The table is
created from the **real columns** the API returns (no guessed schema), all `TEXT`
plus a `_loaded_at` timestamp, with a `UNIQUE(transaction_id)` constraint so
re-runs upsert instead of duplicating.

### How it works (resumable per-date backfill)

- **First run** (empty table): starts at the newest `instance_date` and pulls
  records for that date.
- **Each date** is paged contiguously by `transaction_id` (descending) and
  capped at `DLD_MAX_PER_DATE` records; once a date is full or exhausted, the
  backfill steps **back one day**.
- **Per run** it fetches up to `DLD_RECORDS_PER_RUN` records, then stops.
- **Just re-trigger it** to keep walking backward through history (toward 2012).
  It resumes automatically from the oldest stored date. It stops cleanly when it
  reaches the oldest available data.

### Run it

```cmd
cd backend
.venv\Scripts\python.exe -m scripts.pull_and_store_dld
```

> Use the explicit `.venv\Scripts\python.exe`. On Windows, a plain `python`
> (even with the venv "activated") can resolve to a global interpreter that
> lacks the project dependencies - which fails with `ModuleNotFoundError`.

> Module form (`-m scripts.pull_and_store_dld`) is required, not
> `python scripts/pull_and_store_dld.py`, so the package imports resolve.

### Requirements (`.env`)

| Group | Keys |
|---|---|
| DLD API | `DLD_CLIENT_ID`, `DLD_CLIENT_SECRET`, `DLD_SECURITY_IDENTIFIER`, `DLD_BASE_URL` |
| Supabase Postgres (Session Pooler) | `SUPABASE_DB_HOST`, `SUPABASE_DB_PORT`, `SUPABASE_DB_USER`, `SUPABASE_DB_PASSWORD`, `SUPABASE_DB_NAME` |
| Backfill tuning | `DLD_MAX_PER_DATE` (default 350), `DLD_RECORDS_PER_RUN` (default 500) |

Notes:
- The DDA API is **UAE-only** - run from a UAE connection or it fails at the
  network layer.
- Current credentials are **staging** (`DLD_BASE_URL=https://stg-apis.data.dubai`).
- Connect to Supabase **by parts** (above), not a single URL - the password
  contains special characters that break URL parsing, and the Session Pooler
  host is IPv4-friendly.

### Quota (approved tier)

- **20 requests/minute**, **500 requests/day** - enforced in-process by the
  client's rate limiter. Each run is ~1-2 requests, so you can run it many times
  a day. Up to 1000 records can be returned per request.

### Output

Each run prints the date it's working on, how many new rows it added, the
per-date progress (e.g. `2025-11-04: +350 (now 350/350)`), and a final summary
with the table's total row count and date span.

---

## `pull_and_store_rent_contracts.py` - pull live DLD Ejari rent contracts

The **rental counterpart** of `pull_and_store_dld.py`. Pulls registered rent
contracts from the DDA `dld_rent_contracts-open-api` dataset (Ejari) into the
Postgres table `dld_rent_contracts_raw`. Joined to the sales table on area +
property type, these give the annual rent needed to compute gross rental yield -
the training label for the yield model.

Resumable backfill keyed on the **load batch**, not an event date:

- natural key: `contract_id` (string-sortable, like `transaction_id`)
- **cursor: `load_timestamp`** (the DDA load-batch stamp) - DDA loads rents in
  bulk batches (all rows in a batch share one `load_timestamp`), and
  `contract_start_date` is scrambled on staging, so the load batch is the
  meaningful bucket. Steps back through load batches, capping each at
  `DLD_RENT_MAX_PER_LOAD_TS` (5000) stored rows.
- `MAX_LOAD_TS_PER_RUN` (default **1**) caps how many load batches one run
  processes - raise it (in the script) to walk further back through load history.

It reuses the same DDA credentials, auth, rate limiter (20/min, 500/day), and
the sales script's pure SQL/DB helpers - only the dataset, table, key, and
cursor differ. Tuning knobs: `DLD_RENT_MAX_PER_LOAD_TS`, `DLD_RENT_RECORDS_PER_RUN`.

### Run it

```cmd
cd backend
.venv\Scripts\python.exe -m scripts.pull_and_store_rent_contracts
```

Same `.env` requirements and tuning knobs (`DLD_MAX_PER_DATE`,
`DLD_RECORDS_PER_RUN`) as the sales pull. UAE-only, staging creds.

> **Staging note:** on staging the Ejari text/date fields are scrambled
> placeholders (every `contract_start_date` reads `2030-12-31`), so the per-date
> spread is only meaningful against production data. The pipeline, schema, and
> storage are correct regardless.
