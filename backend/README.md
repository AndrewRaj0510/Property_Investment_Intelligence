# Backend - Property Investment Intelligence Platform

FastAPI service that predicts and explains Dubai gross rental yield.
See the [root README](../README.md) for the full guide. The original spec (PRD,
architecture, plan, tasks) is kept in the author's local `artifacts/` folder, which is
not part of this repository.

## Status

**Live-only architecture (no mock data).** The app uses real data sources only:
Supabase for storage, data.dubai for DLD, OpenAI for LLM steps. It boots without
credentials but data endpoints stay unavailable (health reports `degraded`)
until Supabase is configured. The model is trained **offline, by hand**
(`python -m scripts.train_model`); the only scheduled job is the weekly macro refresh.
The API serves whatever model is already active in Supabase.

## Quick start (local)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows

# Run - launch uvicorn THROUGH the venv's python (see note below):
& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
# macOS/Linux
cd backend
python -m venv .venv
source .venv/bin/activate && pip install -r requirements.txt
uvicorn main:app --reload
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health

Without Supabase the health check returns `{"db": false, "status":
"degraded"}` - expected until credentials are added to `.env`.

> **Windows/OneDrive gotcha:** a bare `uvicorn main:app` resolves to the *system*
> Python (which lacks the deps) and fails with `ModuleNotFoundError: No module named
> 'structlog'`. Always use `& ".\.venv\Scripts\python.exe" -m uvicorn …`, or activate
> the venv first (`.\.venv\Scripts\Activate.ps1`). Avoid `--reload` on Windows here -
> its worker subprocess can re-resolve to the system Python.

## Quick start (Docker)

```bash
cd backend
cp .env.example .env   # then fill in Supabase, OpenAI, DLD, and JWT/admin values
docker compose up
curl http://localhost:8000/api/v1/health
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

## Going live

1. Create a Supabase project; run [`sql/schema.sql`](sql/schema.sql) in the SQL
   editor to create the supporting tables. (The DLD table is created separately
   from the real pull - see below.)
2. Copy `.env.example` → `.env` and fill in `SUPABASE_URL`, `SUPABASE_KEY`,
   `SUPABASE_DB_URL`, `OLLAMA_BASE_URL`/`OLLAMA_MODEL`, the `DLD_*` data.dubai credentials, and
   the security keys (`JWT_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`).
3. Restart - `/api/v1/health` reports `"db": true` once Supabase is reachable.

### Pulling DLD records (live, UAE-only)

The DLD table is created dynamically from the real columns the data.dubai API
returns (no guessed schema). With `DLD_*` + `SUPABASE_DB_URL` set:

```bash
.venv/Scripts/python.exe -m scripts.pull_and_store_dld
```

First run (`DLD_FETCH_LIMIT=1`): creates `dld_transactions_raw` from the pulled
record's columns and inserts it. Later runs only insert (table already exists);
raise `DLD_FETCH_LIMIT` (or set `0`) for a full pull.

## Secured endpoints

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /api/v1/auth/token` | none (login) | Returns a short-lived JWT; rate-limited 5/min; fail-closed |
| `POST /api/v1/admin/ingest/dld` | JWT (admin) | Triggers a live DLD pull; rate-limited 3/min |

## Layout

See the [root README](../README.md#repository-structure) for the full tree. Key files:

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app, CORS, security headers, rate limiter, router wiring, lifespan |
| `config.py` | Pydantic settings; `supabase_configured` / `llm_configured` / `dld_configured` / `jwt_configured` |
| `db/client.py` | Supabase singleton (required; raises if unconfigured) |
| `db/reference.py` | Real Dubai community + centroid reference data |
| `ingestion/dld_client.py` | Live-only data.dubai DLD client (token, pagination, rate/quota limits) |
| `api/security.py` | JWT auth, password hashing, fail-closed admin guard |
| `sql/schema.sql` | Supporting tables (idempotent); DLD table is created by the pull script |
| `scripts/pull_and_store_dld.py` | One-time: create DLD table from real pull + insert records |
```
