# Property Investment Intelligence - Dubai

**Predict and explain the gross rental yield of a Dubai property, and turn a single
pasted listing into an invest-or-avoid verdict - using only real Dubai Land
Department (DLD) sales and Ejari rent data.**

MSc dissertation project · Andrew Isaac Raj G (M01051900) · Middlesex University Dubai
· Supervisor Dr. Siddhaling Urolagin.

- **Backend:** FastAPI (Python) - serves a trained XGBoost yield model + a live
  listing scorer.
- **Frontend:** Next.js 14 / TypeScript ("YieldLens Dubai") - a URL-driven UI that
  scores a pasted Bayut/Dubizzle listing.
- **Data & model:** trained offline on ~2,552 aggregated market segments distilled
  from 117k real sales + 250k real Ejari rent contracts (2012-2026). Reported
  held-out test **R² 0.872 / MAE 0.51 pp**, but see the leakage note below: the
  leakage-free figure is **R² ≈ 0.66**.

> ⚠️ **Accuracy caveat (F45).** Two features, `price_per_sqm` and `rent_per_sqm`,
> are computed from the same medians as the target, so every headline R² in this
> file is inflated by about 0.20. Measured with
> `backend/ml/experiments/run_leakage_check.py`: CV R² 0.865 with them, **0.659**
> without them, **0.663** with them lagged one year (leakage-free). The served
> model has not yet been retrained.

> Deep design rationale and every result (findings **F1-F45**), the full technical
> map and a viva Q&A crib are kept in the author's local `artifacts/` folder
> (`DISSERTATION_FINDINGS.md`, `ARCHITECTURE.md`, `VIVA_PREP.md`), which is not part
> of this repository. Finding IDs below refer to that file.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Scope & coverage - what it can and cannot score](#scope--coverage--what-it-can-and-cannot-score)
3. [Aims & objectives - checklist](#aims--objectives--checklist)
4. [Architecture at a glance](#architecture-at-a-glance)
5. [Repository structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Running the backend](#running-the-backend)
8. [Running the frontend](#running-the-frontend)
9. [Configuration (environment variables)](#configuration-environment-variables)
10. [Data & model lifecycle](#data--model-lifecycle)
11. [API endpoints](#api-endpoints)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)
14. [Design decisions worth knowing](#design-decisions-worth-knowing)

---

## What it does

A property listing only ever states **one half** of the yield ratio - a for-sale
listing has the *price* but not the rent; a for-rent listing has the *rent* but not
the price. This system supplies the **missing half** from a model trained on the whole
official market, computes the listing's gross yield, and benchmarks it against
comparable stock:

```
gross yield = annual rent ÷ sale price
```

- **For a SALE listing:** the model supplies the *expected rent* for that
  (area, type, bedrooms) segment → yield = rent ÷ listing price.
- **For a RENT listing:** the model supplies the *expected sale price* → yield =
  listing rent ÷ price.

The verdict compares the listing's **actual** yield to the model's **expected** yield:
*Good investment* / *Below market* / *Insufficient data*. Every prediction is explained
with SHAP feature attributions and a short LLM-generated narrative.

It is **not** a price predictor, and it never re-predicts facts the listing already
states. See finding **F8** for the full framing.

---

## Scope & coverage - what it can and cannot score

**This is a proof of concept over an ingested window of DLD open data, not a
complete model of Dubai.** The numbers below are measured, not estimated
(finding **F41**):

| | Count |
|---|---|
| Areas with DLD **sales** records (117,443 rows) | 180 |
| Areas with DLD/Ejari **rent** records (250,826 rows) | 182 |
| Areas with **both** - a gross yield needs both sides | 167 |
| **Areas actually scoreable** (`community_yield_medians`) | **127** |
| **Scoreable segments** (area × property type) | **171** |

Dubai Land Department names roughly 226 areas, so this covers ~80% of them and
**~56% are scoreable**.

**Coverage is uneven by property type, and this matters more than the headline:**

| Property type | Areas covered |
|---|---|
| villa | 104 |
| apartment | **66** |
| hotel_apartment | 1 |

**61 areas are villa-only** - an apartment pasted in one of them cannot be
scored even though the area looks "covered". Because apartments dominate portal
listings, effective coverage for a typical pasted listing is nearer the
66-area figure than 127.

Depth is mostly healthy (median 388 sales per segment, max 1,500) but the tail
is thin: **27 of 171 segments have fewer than 30 sales**, 12 have fewer than 30
rent contracts, and the minimum is 1. The `low_sample` flag is set at a
threshold well below 30, so it should **not** be read as "this segment is well
evidenced".

The 40 areas that have both sales and rents yet produce no median row are
industrial and old-town districts (Al Qusais Industrial I-III, Al Goze
Industrial I-III, Ras Al Khor Industrial II-III, Al Buteen, Al Sabkha, Al Ras,
Hatta…) - they have transactions but no residential sales↔rent overlap at the
same property type. That is the aggregation behaving correctly.

**Outside the covered set the app refuses to produce a verdict** rather than
extrapolating - it returns *Insufficient data* and says why (**F38**, **F39**).

> **The dominant cause of "cannot score this" is name resolution, not missing
> data.** Three separate listings failed against communities whose data was
> present all along: `Jumeirah Village Triangle (JVT)` (parenthesised acronym,
> **F36**), the JVC/JVT mapping, and `Costa Brava` - a *cluster* inside DAMAC
> Lagoons, which DLD records as `Al Hebiah Fifth` with 456 villa sales (**F41**).
> Each was an alias/breadcrumb gap. Known remaining gap: a portal that names
> **only** a cluster, with no broader breadcrumb level, still fails; mining
> `master_project_en` (130 master projects) into the alias table would close it.

---

## Aims & objectives - checklist

Status mirrors the author's local `AIM.md`, the authoritative proposed-vs-delivered
record (verified against the live database and codebase **2026-08-10**; objective 2.3
updated **2026-08-24**; Phase-2 and leakage status re-verified **2026-09-28**).
Evidence for every claim is indexed by finding ID in the local findings file.

Legend: **[x]** delivered · **[~]** delivered in a modified form, or partially ·
**[ ]** not implemented.

### Aim 1 - Multi-source data integration pipeline for gross rental yield ✅ achieved

> Develop a multi-source data integration pipeline that enables gross rental yield
> computation at community and property-type level across Dubai.

- [x] **1.1 - DLD sale transactions.** Collect and preprocess DLD sale records to
  establish sale price benchmarks.
  → **117,443** rows in `dld_transactions_raw`; sourced from the **`data.dubai` DDA
  OpenAPI** rather than the Dubai Pulse portal (same publisher; incremental, resumable,
  quota-bounded). Preprocessing is declarative SQL, not pandas (**F16**).
- [~] **1.2 - Bayut/Dubizzle rental listings.** Scrape and normalise long-term rentals
  to derive rental income benchmarks.
  → **Per-URL scraping delivered** for both portals (memo23 Apify actors, live-verified).
  **Bulk community benchmarks deliberately replaced** by DLD Ejari registered tenancy
  contracts (**250,826**) after Bayut proved hard-blocked by DataDome + hCaptcha -
  contracted rents beat asking rents for a yield study. *Objective needs re-wording.*
- [ ] **1.3 - Inside Airbnb short-term rental data.** Compute short-term income
  potential per community.
  → **De-scoped to Future Work (F7)** - a recorded scope decision, taken early. This
  work models long-term yield only; `gross_yield_st` was removed from the matrix, the
  model, and the API.
- [x] **1.4 - OpenStreetMap locational features.** Nearest metro distance and 1 km POI
  density per community.
  → **180 / 180 areas (100%)**, integrated into the feature set and evaluated under
  identical CV - a **measured null result**: +0.001 R² on the production model, inside
  fold-to-fold noise (**F22c**). Reported as a finding about the Dubai data, not a failure.
- [x] **1.5 - LLM extraction of qualitative signals.** Finish tier, view type, amenity
  specificity from listing descriptions.
  → All three live in `POST /api/v1/score` via **OpenAI (gpt-5)**, with self-hosted
  Ollama retained as an offline fallback (**F40**); tolerant parser +
  vocabulary validation, unrecognised values yield `None`, never a guess (**F19**).
  **Context only, never model inputs** - train/serve parity (**F8**).

### Aim 2 - Explainable ML model of gross rental yield ✅ achieved

> Design, train, and evaluate an explainable machine learning model that predicts gross
> rental yield and identifies its principal drivers at community and property-type level.

- [x] **2.1 - Unified feature matrix.** Structural, locational, temporal, and
  qualitative variables.
  → **2,552 rows** at (community × type × bedrooms × year), 71 communities. Structural,
  locational and temporal delivered; **qualitative deliberately excluded** to avoid
  train/serve skew (**F3/F8**). Bedroom grain + per-sqm density lifted CV R²
  **0.654 → 0.825** (**F11**); the per-sqm part of that lift is target leakage
  (**F45**).
- [x] **2.2 - LightGBM model + baseline comparison.**
  → LightGBM **R² 0.786 / MAE 0.667**, benchmarked on identical folds against naive
  baselines: global median R² −0.016, area+type median (the analyst lookup) R² 0.342.
  The production model **more than doubles** that R² and cuts MAE **58.8 %** (**F10**).
- [x] **2.3 - SHAP explainability + validation.**
  → TreeExplainer, **117 features** attributed and persisted to `shap_summary`; drivers
  surfaced per prediction in the API and UI. Validated against **published literature**
  rather than an expert panel (**F26**): rank consistency ρ **0.943**, plus a genuine
  retrodiction of the 1.4 metro null result. *No practitioner panel - a Limitations line;
  ρ is a post-hoc consistency check, never quote it as a blind test.*
- [x] **2.4 - Comparative model evaluation.** LightGBM vs Random Forest vs XGBoost.
  → All three under identical 5-fold CV (seed 42), RF and XGBoost each tuned so none is
  handicapped; selection is programmatic by CV R². **XGBoost 0.864 > LightGBM 0.786 >
  RF 0.777**. Confirmed on an untouched 20 % hold-out: **R² 0.872 / MAE 0.507** (n=511).
  The **winner changed** when the features improved (**F4/F13/F22c**).
  ⚠️ These figures include the per-sqm leakage; without it XGBoost scores **R² 0.659**
  under the same CV, still well above the no-model persistence baseline (0.399) (**F45**).
- [x] **2.5 - Investor dashboard.** Filter by budget, community and property type;
  yield predictions with plain-language, SHAP-driven explanations.
  → All five requested elements present in the Next.js app: community filter, property
  type, **budget slider** (added 2026-08-10, **F24** - real affordability filtering plus
  expected annual rent), live predictions, signed SHAP drivers, and an LLM narrative
  with grounded Q&A.

### Aim 3 - Context-aware extension (macro, environmental, seasonal) ⚠️ 4 of 5

> Extend the baseline yield prediction model by incorporating time-varying
> macroeconomic, environmental, and seasonal features.

**Objectives 3.1 and 3.2 are delivered as ingestion, 3.4 as measurement, 3.5 as the
dashboard; 3.3's dummies are delivered and its tourism half de-scoped on evidence.**
Aim 3 is complete as a body of work - but it delivers **measured negatives, not an
accuracy gain**. **No tourism data has ever been ingested** into the database; the
assessed CSVs are kept under `backend/data/tourism/` as evidence.

**An extended model now exists, and it is no more accurate.** Since 2026-09-12 the database
holds two active runs - `phase1_v1_xgboost` (phase 1) and `phase2_v1_xgboost` (phase 2, 11
macro features, F35) - and `MACRO_FEATURES` is populated. Objective 3.4 first *measured* the
extended feature set without persisting it (**ΔR² −0.0004**, the controlled figure); the
served model corroborates it from a separate run (**−0.0038**, which is inside tuning noise -
the same phase-1 feature set moves 0.0097 between runs, and fold σ(R²) is 0.0099).

**Any figure claiming a Phase-2 improvement would be fabricated** - and equally, Phase 2 is
not measurably *worse*. The defensible claim is **no improvement**. What the extended model
does show is that the external features carry **3.6%** of its attribution while adding no
predictive power: at annual grain each is a function of `year`, which the baseline already
has. **Attribution is not contribution.**

`activate_run` is phase-scoped, so both models serve simultaneously and **`/score` still uses
the Phase-1 baseline**. Phase 2 is reached deliberately via `use_extended_model=true` or
`GET /api/v1/models/comparison`.

- [x] **3.1 - Macroeconomic ingestion** (EIBOR 3m, UAE CPI, Dubai GDP, Brent).
  → **All four series ingested** into `macro_timeseries` (177 monthly rows, 2012-01 →
  2026-09) and `macro_annual` (15 rows). EIBOR is **1,964 trading days** from a *manual
  browser export* - centralbank.ae is Cloudflare-walled, the **second** official UAE source
  in this project blocked by bot protection after Bayut, which is why it only reaches back
  to 2018-07. Brent from EIA (`RBRTE`, 15/15 training years); UAE GDP/inflation/CPI index
  from World Bank. Three source deviations vs the proposal (manual export not CBUAE fetch;
  World Bank not UAE FSC; EIA not FRED) and UAE-level, not Dubai-level, GDP.
  ⚠️ **No retrain, by design.** At the Phase-1 join grain every macro feature collapses to
  a function of `year` - 9-15 distinct values across 2,552 rows - and `year` is already a
  model feature, so none of them can carry information the model cannot already reach.
  Same failure mode as the 1.4 OSM null, but **predicted from the grain and then
  confirmed** rather than measured after the fact. Run
  `python -m scripts.ingest_macro --dry-run` to print the evidence.
- [x] **3.2 - Environmental data + thermal discomfort index.**
  → **5,362 daily observations** from the Open-Meteo ERA5 archive (keyless), 2012-01-01 →
  2026-09-05, **zero nulls**, every calendar year complete - the only series with
  **15/15** training-year coverage. No source deviation: the objective named Open-Meteo.
  ⚠️ **The specified index formula was dimensionally wrong** - it used relative humidity
  (%) where the BoM apparent-temperature equation wants vapour pressure (hPa),
  overstating by a mean of 9 °C and **halving the seasonal spread (24.6 → 13.6 °C)**. It
  agrees within 1 °C on the hottest day, so a spot check would have passed it - the third
  such plausible-but-wrong defect in this project (F29). Corrected via Magnus-Tetens.
  ⚠️ **No retrain, by design - and here the null is measured, not just argued:**
  **99.4% of the thermal-index variance is destroyed** by aggregating to the annual join
  grain (seasonal σ is **12.9×** inter-annual σ). Dubai weather does not vary between
  years, it varies within them.
- [~] **3.3 - Tourism volumes + seasonal calendar dummies.** *(dummies delivered; tourism de-scoped)*
  → **Seasonal dummies delivered.** Ramadan is now **computed from the Hijri calendar**,
  replacing a hardcoded table that covered only **7 of the 29** months containing Ramadan
  days, left **8 of 15** training years unflagged, and for 2020 and 2023 flagged the
  *wrong* month (Ramadan 2020 had 7 days in April and **23 in May** - it flags April).
  The **COVID-19 break - previously absent from the codebase entirely** - is now encoded
  both as an acute pulse and as a permanent structural step, alongside the Expo window.
  ⚠️ **Tourism volumes formally DE-SCOPED on measured evidence (F31).** Twelve DSC/DET
  files were obtained and assessed: the actual international-visitor series covers **3 of
  15** training years, hotel occupancy is **quarterly not monthly** and covers 9 of 15
  with **2020 missing**, and the only full-coverage monthly series (DXB airport arrivals)
  is a **proxy**, not tourist arrivals. Rather than discard it untested, the best
  available proxies were joined and measured: **ΔR² −0.0033** - inside fold noise. No
  production tourism ingestion was built; the CSVs are kept under `backend/data/tourism/`
  as evidence.
- [x] **3.4 - Retrain with the extended set + time-based splits.** *(as measurement)*
  → **The extended feature set does not help: ΔR² −0.0004** on the production model against
  a fold-to-fold sd of 0.0142 - 0.03× the noise, and *negative* for all three models. That
  completes the arc F28 predicted and F29 quantified. **Time-based CV also run**: random
  5-fold **0.8202** vs forward chaining **0.6926**. An isolation test (same test rows, same
  training-set *size*) attributes only **39%** of that 0.128 gap to leakage - and that
  estimate is not statistically solid (sd 0.106, two years negative). The other **61%** is
  the intrinsic difficulty of predicting an unseen future year (F30).
  ⚠️ `TimeSeriesSplit` is **not** adopted as the production protocol.
  A Phase-2 model **is** now persisted and active (`phase2_v1_xgboost`, 2026-09-12, F35) with
  `MACRO_FEATURES` populated - it measures no accuracy gain.
  `backend/artefacts/phase2_v2_lightgbm.joblib` and `phase1_v2_lightgbm.joblib` remain
  **mock-era leftovers - do not cite them**; the live artefacts are the `*_v1_xgboost` pair.
- [x] **3.5 - Macro overlay, seasonal calendar, two-level SHAP.**
  → **Overlay panel built** (`MacroOverlay.tsx`, opened from the header): EIBOR, Brent,
  max temp and thermal index on real data, each with its own **as-of month**, and the
  panel states in the UI that these are **context, not model inputs**.
  → **Seasonal calendar built - as a CLIMATE calendar, deliberately.** There is no
  monthly yield in this project (the matrix is annual), so a seasonal *yield* calendar
  would be invented. The old `yield_index` was **removed rather than left null**; it is
  replaced by thermal discomfort indexed to the annual mean (Jan **66.6** → Jul
  **128.6**), and the endpoint ships `basis`/`note` in the payload so the chart cannot be
  relabelled.
  → **Two-level SHAP un-gated.** It was gated on `phase == 2` and had therefore never
  run. It now returns **100% property / 0% macro** on every prediction - *the Aim 3
  result rendered*, with `macro_note` explaining that the macro series were ingested,
  evaluated and measured at ΔR² −0.0004 (F32).
  → **The last fabricated numbers are gone**: `shapRows()` no longer falls back to
  `DEMO_SHAP`, which advertised "Metro proximity" and "Sea view" - precisely what this
  project measured as null (F22c) and deliberately excludes (F3/F8). That was F27 U4.
  **The dashboard no longer fabricates a single displayed number.**

### Beyond the numbered objectives

- [~] **Dashboard usability evaluation** (proposal §5.4). Heuristic inspection done:
  **10 findings, 3 at severity 4** (**F27**, local `USABILITY_EVALUATION.md`). No
  participants tested; the task protocol is written but unrun.

**Overall: Aim 1 achieved** (4 of 5 objectives delivered, 1 de-scoped with a recorded
rationale, 1 delivered in a modified form that is defensibly superior to the original
plan) · **Aim 2 achieved** (5 of 5) · **Aim 3 complete as a body of work, 4 of 5 + half**
(3.1 and 3.2 ingested, 3.4 measured, 3.5 built; 3.3's dummies delivered and its tourism
half de-scoped on measured evidence) - **but delivering measured negatives, not an
accuracy gain.** A Phase-2 model is trained and active (`phase2_v1_xgboost`, CV R²
0.8598 against Phase 1's 0.8636), and it is no more accurate.

> **A note on the Aim 3 negatives.** They are not separate failures. They are one finding
> measured four times: *the grain of the join, not the quality of the feature, decides
> whether an external series can inform a model* - and it is checkable in advance.
> F28 predicted it, F29 quantified it (99.4% of weather variance lost to annual
> aggregation), F30 measured the consequence (ΔR² −0.0004), and F31 confirmed it again on
> tourism (ΔR² −0.0033) - which is why that half of 3.3 is de-scoped on evidence rather
> than on impression. **F32 then put it on screen**: the dashboard's two-level SHAP split
> reads 100% property / 0% macro, with the reason attached.

---

## Architecture at a glance

```
                 ┌──────────────────────────────────────────────┐
   Live listing  │              FRONTEND (Next.js 14)            │
   URL pasted →  │  YieldLens Dubai · TS · same-origin proxy     │
                 └───────────────────────┬──────────────────────┘
                                          │  /api/v1/*  (Next rewrite → FastAPI)
                 ┌────────────────────────▼──────────────────────┐
                 │                 BACKEND (FastAPI)              │
                 │                                                │
                 │  API routes  ── /score  /predict  /communities │
                 │              ── /compare /shap /health  /auth   │
                 │                                                │
                 │  services/listing_scorer  ← the product core    │
                 │       │                                         │
                 │       ├── ingestion/listing_client (Apify)      │
                 │       ├── ml/predict  (XGBoost + SHAP)          │
                 │       └── db/queries/yield_agg (median baseline)│
                 │                                                │
                 │  llm/ (LangChain + OpenAI: narrate/classify/QA) │
                 │  scheduler/ (weekly macro refresh, Mon 03:00)   │
                 └────────────────────────┬──────────────────────┘
                                          │  psycopg (direct Postgres)
                 ┌────────────────────────▼──────────────────────┐
                 │            Supabase PostgreSQL                 │
                 │  dld_transactions_raw · dld_rent_contracts_raw │
                 │  feature_matrix · community_yield_medians      │
                 │  model_runs · shap_summary                     │
                 └────────────────────────────────────────────────┘
                                          ▲
     OFFLINE (manual, run as scripts):  DLD DDA OpenAPI → ingest → clean (SQL)
                                        → feature matrix → train → persist model
```

**Two independent flows:**

1. **Offline training** (`scripts/train_model.py`, run by hand; only the macro refresh is
   scheduled): pull DLD data → build the
   cleaned feature matrix in SQL → compare LightGBM/RF/XGBoost under 5-fold CV → persist
   the winning model + SHAP summary to Postgres. **Not run on boot.**
2. **Online serving** (the live API): the app loads the already-trained model and
   scores live listings / prediction requests. Fails loud if the datastore is
   unconfigured - never fabricates data.

> **Note on the datastore:** the Supabase **ORM** client is *not* configured in this
> deployment (no `SUPABASE_URL`/`KEY` - only the `SUPABASE_DB_*` connection parts).
> All DB access runs through a **direct psycopg connection** to pooled Postgres.
> See finding **F6**.

---

## Repository structure

```
Property_Investment_Intelligence/
├── README.md                     ← you are here
├── .gitignore                    ← also excludes the local-only notes below
│
├── backend/                      ← FastAPI service (see detail below)
│   └── README.md                 ← backend-specific quick start
│
└── frontend/                     ← Next.js 14 / TypeScript app ("YieldLens Dubai")
    └── README.md                 ← frontend-specific notes
```

Kept **locally and excluded from the repository** by `.gitignore`: `artifacts/`
(dissertation drafts, findings F1-F45, architecture notes, viva prep, DLD docs),
`AIM.md` (aims tracker), `Log.md` (activity log), `MEMORY.md`, `.claude/`, and
`backend/.env`.

### Backend (`backend/`)

```
backend/
├── main.py                 # FastAPI app: routers, CORS, security headers, rate limiter, lifespan
├── config.py               # Pydantic settings + *_configured flags
├── logging_config.py       # structlog setup
├── requirements.txt        # pinned dependencies
├── pytest.ini              # test config (pythonpath=., asyncio auto)
├── Dockerfile, docker-compose.yml
├── README.md
├── sql/
│   └── schema.sql          # supporting tables (idempotent). Raw DLD tables are created by the pull scripts.
│
├── api/                    # HTTP layer
│   ├── routes/             # score, predict, explain, communities, compare, comparison, shap, market, macro, seasonal, health, auth, admin
│   ├── schemas/            # Pydantic request/response models
│   ├── security.py         # JWT auth, bcrypt password hashing, fail-closed admin guard
│   └── rate_limit.py       # SlowAPI limiter
│
├── services/
│   └── listing_scorer.py   # ★ product core: fetch listing → model expected yield → verdict
│
├── ml/                     # model training & inference (all via psycopg)
│   ├── train.py            # 3-model comparison, RF+XGB tuning, empirical selection by CV R²
│   ├── encoding.py         # SHARED feature encoding (train == serve column parity)
│   ├── predict.py          # inference: predict_yield_for() + SHAP; community/type aliasing
│   ├── baseline.py         # naive baselines (global & area+type median)
│   ├── evaluate_holdout.py # 80/20 held-out eval + residual analysis
│   ├── persist.py          # psycopg readers/writers for the model registry & feature matrix
│   ├── shap_explainer.py   # SHAP TreeExplainer wrapper
│   ├── artefacts.py        # joblib save/load
│   ├── macro_features.py   # Phase-2 macro join (annual grain)
│   ├── community_resolve.py # portal name → DLD area_name_en via mined aliases
│   └── experiments/        # isolated, read-only experiments (F11 features, OSM, Phase 2, SHAP, F45 leakage)
│
├── db/
│   ├── pg.py               # psycopg connection + identifier quoting (the path actually used)
│   ├── client.py           # DB health ping + optional Supabase Storage for model upload (ORM unused)
│   ├── reference.py        # real Dubai community + centroid reference data
│   └── queries/
│       ├── feature_matrix_dld.py  # ★ builds the cleaned training matrix in SQL (F16)
│       ├── yield_agg.py           # sales↔rents median yield baseline (community_yield_medians)
│       └── community_aliases.py, community_features.py, macro.py, price_history.py
│
├── ingestion/              # data-source clients (live-only, fail closed)
│   ├── dld_client.py            # DLD via data.dubai DDA OpenAPI (token, paging, quota); sales by default
│   ├── dld_rent_client.py       # the same client pointed at the Ejari rent-contracts dataset
│   ├── listing_client.py        # ★ per-URL Bayut/Dubizzle fetch via Apify actors
│   ├── listing_fixtures.py      # capture/replay of listings (LISTING_FIXTURE_MODE)
│   ├── osm_client.py            # OpenStreetMap metro distance + POI density
│   ├── eibor_client.py, eia_client.py, worldbank_client.py, openmeteo_client.py   # macro + weather (3.1/3.2)
│   └── seasonal_calendar.py     # Hijri-computed Ramadan, COVID and Expo dummies (3.3)
│
├── llm/                    # LangChain chains + prompts; LLM_PROVIDER picks OpenAI (default) or Ollama; OPENAI_MODEL=gpt-5 here (code default gpt-4o-mini)
│   ├── provider.py              # OpenAI / Ollama factory, reasoning-token headroom
│   └── chains/                  # explain_score (/score narrative), explain_shap (/predict narrative),
│                                # classify_listing (finish/view/amenities), answer_question (/explain)
│
├── scheduler/              # APScheduler: weekly macro refresh only (Mondays 03:00)
│   ├── jobs.py             # schedule definition
│   └── tasks.py            # refresh_macro
│
├── scripts/                # operational entrypoints (run as modules) - THE pipeline
│   ├── train_model.py           # ★ full offline training run → persists active model (manual)
│   ├── pull_and_store_dld.py    # create + backfill dld_transactions_raw from live pulls
│   ├── pull_and_store_rent_contracts.py
│   ├── refresh_yield_medians.py # rebuild community_yield_medians baseline
│   ├── refresh_price_history.py # rebuild community_price_history (results-screen chart)
│   ├── refresh_osm_features.py, refresh_community_aliases.py, ingest_macro.py
│   ├── capture_listing.py       # save a listing for offline replay
│   └── load_*_csv.py            # bulk CSV loaders (alternative to live pulls)
│
└── tests/                  # pytest: API, security hardening, ingestion, LLM chains, scoring (323 tests)
```

### Frontend (`frontend/`)

```
frontend/
├── package.json            # scripts: dev / build / start / lint
├── next.config.mjs         # ★ rewrites /api/v1/* → FastAPI (same-origin proxy)
├── tsconfig.json
└── src/
    ├── app/
    │   ├── page.tsx         # ★ main flow: idle hero → loading → results, or AnalysisFailed
    │   ├── layout.tsx
    │   └── globals.css
    ├── components/          # IdleHero, LoadingScreen, ResultsScreen, AnalysisFailed, ListingMap,
    │                        # PriceTrendChart, ComparisonPanel, MacroOverlay, ScenarioModal, AppHeader, …
    └── lib/
        ├── api.ts           # ★ backend client (scoreListing, getMacroSnapshot, …); failures are reported, never faked
        ├── types.ts         # TS mirrors of the API response schemas
        ├── labels.ts        # SHAP feature-name → friendly label
        ├── listingSource.ts # Bayut vs Dubizzle URL validation
        ├── demo.ts          # loading-step labels and scenario-modal constants
        └── useTheme.ts
```

★ = the files most worth reading first.

---

## Prerequisites

| Tool | Version | For |
|---|---|---|
| Python | 3.11+ | backend |
| Node.js | 18.17+ (Next 14) | frontend |
| A Supabase Postgres | - | data + model store (or any reachable Postgres) |
| OpenAI API key | - | SHAP narrative + listing classification (optional to boot) |
| Apify token | - | live listing fetch (optional to boot; without it `/score` returns 503 and the UI says why) |

The app **boots without any credentials** - data/ML endpoints simply report degraded
or return errors until the relevant service is configured. Nothing is faked.

---

## Running the backend

```powershell
cd backend

# 1. create the virtualenv (first time only)
python -m venv .venv

# 2. install dependencies (first time only)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. configure (optional to boot, required for live data - see next section)
cp .env.example .env    # then edit .env

# 4. run - Windows / PowerShell (the reliable command on this project):
& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- API root: <http://localhost:8000>
- Interactive docs (Swagger): <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/v1/health>

`/api/v1/health` pings Postgres over the **psycopg** path (`SUPABASE_DB_*` parts), which
is what every data endpoint uses; it reports `"db": false` / `"degraded"` only when those
parts are missing or the database is unreachable. A `supabase_not_configured` log line
refers to the unused ORM `SUPABASE_URL`/`KEY` and is informational.

> **Always launch uvicorn through the venv's own Python** -
> `& ".\.venv\Scripts\python.exe" -m uvicorn …` - not a bare `uvicorn main:app`. On this
> Windows/OneDrive setup a bare `uvicorn` resolves to the *system* Python (which lacks the
> dependencies) and fails with `ModuleNotFoundError: No module named 'structlog'`. The
> `-m uvicorn` form guarantees the venv interpreter. Avoid `--reload` here: its worker
> subprocess can re-resolve back to the system Python - restart the server manually after
> code edits instead.

**macOS / Linux equivalent:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate           # bare `uvicorn` is fine once activated
pip install -r requirements.txt
uvicorn main:app --reload
```

**Optional - activate the venv (PowerShell)** so a bare `uvicorn` resolves correctly for
the whole session (look for `(.venv)` in your prompt):

```powershell
.\.venv\Scripts\Activate.ps1
# if blocked by execution policy (session-only, safe):
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Docker (alternative)

```bash
cd backend
cp .env.example .env      # fill in credentials
docker compose up
curl http://localhost:8000/api/v1/health
```

---

## Running the frontend

The frontend proxies all `/api/v1/*` calls to the backend via a Next.js rewrite, so the
browser talks **same-origin** and CORS never bites.

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

By default the rewrite target is `http://127.0.0.1:8000` (a locally-running backend).
Override it if your backend is elsewhere:

```bash
API_PROXY_TARGET=http://your-backend:8000 npm run dev
```

Other scripts: `npm run build` (production build + type-check), `npm run start` (serve
the build), `npm run lint`.

**Run order for the full experience:** start the **backend** first (so the proxy target
is live), then the frontend. If the backend is down or a listing can't be scored, the UI
shows an **analysis failed** screen with the reason; it never substitutes demo figures
(F38). To demonstrate offline, use a captured listing (`LISTING_FIXTURE_MODE=replay`,
below).

---

## Configuration (environment variables)

Backend config is loaded from `backend/.env` (see `backend/.env.example`). Grouped by
what each unlocks:

### Datastore (required for any real data)
```env
# The psycopg path actually used - connect by PARTS (avoids URL-encoding a '@' in the
# password). Use the Supabase Session Pooler values (IPv4-friendly host).
SUPABASE_DB_HOST=aws-1-…pooler.supabase.com
SUPABASE_DB_PORT=5432
SUPABASE_DB_USER=postgres.<project-ref>
SUPABASE_DB_PASSWORD=…
SUPABASE_DB_NAME=postgres

# ORM path (currently NOT used by this deployment; leave blank unless you wire the ORM):
SUPABASE_URL=
SUPABASE_KEY=
```

### LLM (narratives, listing classification, grounded Q&A)
```env
LLM_PROVIDER=openai                      # 'openai' (default) or 'ollama'

# --- OpenAI (the active provider, F40) ---
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5
OPENAI_REASONING_EFFORT=low              # reasoning models only; see below
OPENAI_BASE_URL=                         # only for an Azure/proxy endpoint

# --- Ollama (retained offline / zero-cost fallback, F19) ---
OLLAMA_BASE_URL=http://localhost:11434   # any reachable Ollama server
OLLAMA_MODEL=gpt-oss:120b-cloud          # any model that server has
OLLAMA_API_KEY=                          # only if behind an auth proxy
OLLAMA_ENFORCES_SCHEMA=false             # true only for LOCAL models - see below
```
> Switching provider is one env var; no chain knows which backend is active.
> The project ran entirely on self-hosted Ollama for its measured phase
> (**F19**) and that path still works - point `OLLAMA_BASE_URL` at a local
> daemon, a remote host, or a tunnel and set `LLM_PROVIDER=ollama`.
>
> **Reasoning models need token headroom.** gpt-5 and the o-series spend
> completion tokens *thinking* before emitting any content, and that spend
> counts against the same ceiling - a budget sized for the visible answer
> returns an empty string. Measured: a 200-token ceiling produced nothing at
> all. The factory therefore adds `OPENAI_REASONING_HEADROOM_TOKENS` (default
> 2000) on top of each call site's budget (**F39**, **F40**).
>
> **`OPENAI_REASONING_EFFORT`** defaults to `low`. The optional /score
> enrichment runs on a 20s per-chain budget and degrades *silently* to no
> narrative when it overruns; measured on the live chain, `medium` took
> 11.3-15.2s (one run hit 19.6s) against `low` at 5.7-6.4s - with fuller
> prose. Sent only to reasoning families; `gpt-4o` rejects the parameter.
>
> **`OLLAMA_ENFORCES_SCHEMA`** - Ollama's JSON-schema `format` constraint is applied
> by the *local* inference engine. A server that proxies to a hosted backend (the
> `*-cloud` model tags) accepts the parameter and silently ignores it, returning
> prose instead of JSON. Set this to `false` for such an endpoint: listing
> classification then relies on its tolerant parser, which recovers fields from
> JSON, fenced JSON, or labelled prose and returns `null` for anything it cannot
> validate rather than guessing.
>
> The SHAP narrative on `/api/v1/predict` remains **live-only** (no template
> fallback): it returns **503** if the model runs but the LLM is unset - the
> prediction is fine, only the narrative needs it (**F15**). `/score` behaves
> differently on purpose: its narrative degrades to `null` rather than discarding
> a completed analysis (**F18**).

### Listing fetch mode (test the scoring flow without a live URL)
```env
LISTING_FIXTURE_MODE=live                # "replay" = resolve ANY url to a saved capture
```
> Scoring normally costs a paid Apify fetch per call and needs a live portal URL.
> Capture a listing **once**, then replay it forever, offline and free (**F19**):
> ```bash
> python -m scripts.capture_listing "https://www.bayut.com/property/details-1234.html" --name marina_2br
> python -m scripts.capture_listing --list
> # then score it with:  {"url": "fixture://marina_2br"}
> ```
> A `fixture://<name>` URL works in either mode; `LISTING_FIXTURE_MODE=replay` makes
> *every* URL resolve to a capture so the real frontend runs end-to-end offline.
> Only the network fetch is replaced - the model, DLD baseline, verdict and LLM all
> run as in production. Every response carries **`source_mode`** (`live` | `fixture`)
> so a fixture-backed score is never mistaken for a live one.

### DLD data source (offline training data)
```env
DLD_CLIENT_ID=…
DLD_CLIENT_SECRET=…
DLD_SECURITY_IDENTIFIER=…       # custom DDA integration header
# Quota is enforced in-client: 20 req/min, 500 req/day.
```

### Live listing scoring (the `/score` flow)
```env
APIFY_TOKEN=…
# actors default to the memo23 Bayut + Dubizzle scrapers (accept a detail URL)
```

### API security (secured admin endpoints)
```env
JWT_SECRET=…                    # no default - secured endpoints fail closed if unset
ADMIN_USERNAME=…
ADMIN_PASSWORD_HASH=…           # bcrypt hash, never plaintext
CORS_ALLOWED_ORIGINS=*          # set to your frontend origin(s) in production
```

### Frontend
```env
# frontend/.env.local  (see frontend/.env.example - optional)
API_PROXY_TARGET=http://127.0.0.1:8000    # where the Next rewrite proxies /api/v1/*
NEXT_PUBLIC_API_BASE=/api/v1              # override only to bypass the proxy
```

---

## Data & model lifecycle

Everything is **offline and run by hand**, never on boot. The only scheduled job is the
weekly macro refresh (Mondays 03:00, `scheduler/jobs.py`).

**1. Ingest raw DLD data** into Postgres (creates the raw tables from the real columns
the API returns - no guessed schema):
```bash
cd backend
python -m scripts.pull_and_store_dld              # sales → dld_transactions_raw
python -m scripts.pull_and_store_rent_contracts   # Ejari rents → dld_rent_contracts_raw
```
> Run scripts **as modules** (`python -m scripts.…`) from the `backend/` dir so package
> imports resolve.

**2. Build the median baseline** used by the verdict, and the price-history
snapshot behind the results-screen chart:
```bash
python -m scripts.refresh_yield_medians           # → community_yield_medians
python -m scripts.refresh_price_history           # → community_price_history
```

**3. Train the model** (one entrypoint does the whole run):
```bash
python -m scripts.train_model
```
This rebuilds the cleaned feature matrix in SQL, evaluates naive baselines, compares
LightGBM / Random Forest / XGBoost under 5-fold CV, selects the winner by CV R², fits it,
computes the SHAP summary, and persists **one active run** to `model_runs`.

**How the data is cleaned (finding F16).** There is no separate pandas cleaning script -
cleaning is **declarative SQL** in `db/queries/feature_matrix_dld.py`: row-level validity
filters (regex numeric-parse, valid year, parseable bedrooms), canonicalisation of the two
type vocabularies, median aggregation to `(area × type × bedrooms × year)`, an inner join
of sales↔rents, and trustworthiness gates (≥3 sales AND ≥3 rents; yield ∈ 1-20%). The raw
117k sales / 250k rents distil to **2,552 training rows** - driven by aggregation + the
inner join, not by scrubbing dirty data.

---

## API endpoints

All under `/api/v1`. Full interactive contract at `/docs`.

| Method & path | Purpose | Auth |
|---|---|---|
| `GET /health` | Liveness + datastore status | none |
| `POST /score` | **Score one pasted Bayut/Dubizzle listing URL** → verdict | none |
| `POST /predict` | Predict yield for an explicit (community, type, beds, size) | none |
| `GET /communities` | Community yield summary list | none |
| `GET /compare` | Compare communities | none |
| `GET /shap/summary` | Global SHAP driver ranking for the active model (add `?use_extended_model=true` for Phase 2) | none |
| `GET /models/comparison` | **Phase 1 (baseline) vs Phase 2 (+ external factors)** - metrics, delta, macro attribution share (Aim 3) | none (60/min) |
| `POST /explain` | **Ask a question about a `/score` result** - answered strictly from that analysis | none (20/min) |
| `GET /market/price-history` | **Quarterly transacted price history** for one (community, property type), from DLD sales. History only, never a forecast | none (60/min) |
| `GET /macro/snapshot` | Latest ingested macro indicators (EIBOR, Brent, weather), each with its as-of month; context, not model inputs | none |
| `GET /seasonal/calendar` | Monthly **climate** calendar (thermal index, Ramadan days); not a yield calendar | none |
| `POST /auth/token` | Login → short-lived JWT | none (rate-limited 5/min) |
| `POST /admin/ingest/dld` | Trigger a bounded incremental DLD sales top-up into `dld_transactions_raw` (does not retrain) | **JWT (admin)**, 3/min |

**`/score` error mapping:** unsupported site → `422`, Apify unconfigured → `503`,
listing unreadable/blocked → `502`, no comparable baseline → `422`.

**LLM layer on `/score` (see DISSERTATION_FINDINGS F18).** A scored listing also
carries `narrative` (a plain-English explanation of the verdict) and `character`
(LLM-read finish tier / view / amenity score). Two rules matter when consuming them:

* `character` is **context only** - those attributes are not features of the yield
  model (the DLD/Ejari training data has no such columns), so never render them as
  drivers of the predicted yield.
* `narrative` is **optional and may be `null`** when the LLM is unset or
  the call fails. The numeric verdict is never withheld for a missing narrative -
  fall back to `verdict.reason`, which is always present.
* `narrative_detail` carries the same explanation with structure (`intro`,
  `points[{label, detail}]`, `action`); `narrative` is its flattened string.
* The narrative call has its OWN time budget, `OLLAMA_NARRATE_TIMEOUT_S`
  (default 60s), separate from `OLLAMA_ENRICH_TIMEOUT_S` (default 20s, the
  classification call). Do not merge them: the narrative measures ~27s and
  sharing the smaller budget made every live call time out into the `null` path
  above, which looks exactly like an unconfigured LLM (F44).

**`/explain` error mapping:** empty/oversized question or malformed context → `422`,
LLM unset → `503`, generation failure → `502`. It answers only from the
analysis you pass it and refuses when the data doesn't cover the question.

**What `/explain` can see** (F44). Its context is the analysis you post, so what
it can answer is decided by what you send:

* `context` - a `/score` result. Its `baseline` now carries the bucket's
  recorded **spread** (p25/p75/p90 and the maximum, for both sale price and
  annual rent) alongside the medians, so range questions ("the highest rent in
  this community") are answerable. The maximum is a RAW maximum and can be a
  bulk or whole-building transaction, so answers lead with p90 and label it.
* `price_history` (optional) - the series from `GET /market/price-history` that
  the client is drawing beside the analysis. Without it, questions about how
  prices have moved are refused, because the chart is a different endpoint and
  the model cannot see it.

Grounding is unchanged: a question the context does not cover still gets a
refusal naming what is missing, and a request for a forecast is still refused
even with the price series in context.

### Example - score a listing
```bash
curl -s -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.bayut.com/property/details-8214402.html"}' | jq
```

### Example - direct prediction
```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"community":"Dubai Marina","property_type":"Apartment","bedrooms":2,"floor_area_sqft":1100}' | jq
```

---

## Testing

```bash
cd backend
.venv\Scripts\python.exe -m pytest        # Windows
# python -m pytest                        # macOS/Linux (venv active)
```

Current status (2026-09-28): **323 passed, 0 failed** with the database reachable. LLM
and live-fetch calls are mocked by the conftest fixtures, so the suite never hits a paid
API. A few API tests read the live database; with it unreachable they fail with a
psycopg connection error rather than a code fault.

Frontend type-check / build:
```bash
cd frontend && npm run build
```

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `/api/v1/health` shows `"db": false` | Supabase not configured - add `SUPABASE_DB_*` to `backend/.env`. |
| `/predict` returns **503 "…requires an LLM"** | Model works; set `OPENAI_API_KEY`/`OPENAI_MODEL` (or `LLM_PROVIDER=ollama`) for the SHAP narrative. |
| `/score` returns a verdict but `narrative: null` | The LLM is unreachable, rejected the key, or overran its narrative budget (`OLLAMA_NARRATE_TIMEOUT_S`, default 60s). The numbers are still valid (F18, F44). |
| `character` is always `null` | Check logs for `listing_classify_unparsed`. An **empty** `sample=` means the completion ceiling was consumed by reasoning tokens - raise `OPENAI_REASONING_HEADROOM_TOKENS` (F39/F40). On Ollama, set `OLLAMA_ENFORCES_SCHEMA=false` (F19). |
| `/score` returns **503** | `APIFY_TOKEN` unset - live listing fetch is disabled. |
| `/score` returns **502** | The listing couldn't be read (blocked/removed/bad URL); try another. |
| Frontend shows "analysis failed" | The backend was unreachable or `/score` returned an error; the screen states the reason. Check the backend is running on the proxy target. |
| `ModuleNotFoundError` running a script | Run it as a module from `backend/`: `python -m scripts.<name>`. |
| venv not found on OneDrive path | Invoke explicitly: `.venv\Scripts\python.exe …`. |
| Admin endpoint returns 503 | `JWT_SECRET` / `ADMIN_*` unset - secured endpoints fail closed by design. |

---

## Design decisions worth knowing

These are the choices most likely to be questioned - each is documented in the local
findings file:

- **Honest, un-imputed target (F1).** Yield is computed from real medians on both sides;
  segments without both real sales *and* real rents are dropped rather than imputed.
- **Segment grain, not transaction grain (F1/F2).** There's no key to pair an individual
  sale with an individual rent, so the unit of analysis is the market segment.
- **Train/serve feature parity (F3).** The model uses only features present in the
  training data; listing-only attributes (finish, floor, view, amenities) are context /
  narrative, never model inputs.
- **Empirical model selection (F4/F11).** LightGBM/RF/XGBoost compared under identical CV;
  the winner is chosen by CV R² programmatically. XGBoost wins on the engineered features.
- **Feature engineering, with a leakage correction (F11, F45).** Bedroom grain + per-sqm
  density raised R² from 0.654 → 0.825, and tuning took the served model to CV R² 0.864 /
  held-out 0.872. The per-sqm features share the target's own medians, so the
  leakage-free CV R² is **≈ 0.66**; retraining without them is the outstanding fix.
- **psycopg, not the Supabase ORM (F6).** The ORM isn't configured; all DB access is raw
  parameterised SQL over pooled Postgres.
- **Cleaning is declarative SQL (F16).** Reproducible by construction; no manual edits.
- **Fails loud, never fabricates (throughout).** Missing credentials disable endpoints;
  the system refuses to score when it lacks comparable data ("Insufficient data").

> **Phase 2 is built and measured, not frozen.** Macro and weather series are ingested,
> the extended model (`phase2_v1_xgboost`) is trained and active alongside Phase 1, and it
> adds no accuracy. `/score` uses the Phase-1 model; Phase 2 is reached via
> `use_extended_model=true` or `GET /api/v1/models/comparison`.
