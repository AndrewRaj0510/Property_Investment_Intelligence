# YieldLens Dubai - Frontend

Next.js (App Router) + TypeScript + React implementation of the **YieldLens Dubai**
design (imported from the Claude Design project *Dubai Property Investment Platform*).
It is the investor-facing frontend for the Property Investment Intelligence backend.

## Flow

Choose the portal (Bayut or Dubizzle) → paste a listing URL → staged loader while the
backend fetches and scores it → results dashboard (listing card, map, yield gauge and
verdict, structured narrative, SHAP drivers, recorded price history, grounded Q&A) →
optional **Run my scenario** modal for a what-if prediction.

If the analysis cannot be completed, an **analysis failed** screen states the reason.
The UI never substitutes demo figures for a failed or missing result (F38).

Dark theme by default with a light theme toggle (persisted to `localStorage`).

## Backend wiring

API calls go through Next's rewrite proxy at `/api/v1/*` → the FastAPI backend
(default `http://127.0.0.1:8000`, override with `API_PROXY_TARGET`). Endpoints used:

| Where                       | Endpoint                          |
| --------------------------- | --------------------------------- |
| Analyse a pasted listing    | `POST /api/v1/score`              |
| Ask about the result        | `POST /api/v1/explain`            |
| Price history chart         | `GET /api/v1/market/price-history` |
| Header macro strip          | `GET /api/v1/macro/snapshot`      |
| Macro overlay               | `GET /api/v1/models/comparison`, `GET /api/v1/seasonal/calendar` |
| Scenario modal              | `GET /api/v1/communities`, `POST /api/v1/predict` |

`scoreListing` returns either the result or a human-readable reason (rate limit,
unconfigured service, unreadable listing, timeout, backend unreachable). The other
calls return `null` on failure and the affected panel shows that the data is
unavailable.

To demonstrate without a live portal fetch, run the backend with
`LISTING_FIXTURE_MODE=replay` (see the root README).

## Run

```bash
npm install
npm run dev        # http://localhost:3000  (proxies /api/v1 → 127.0.0.1:8000)
```

Start the FastAPI backend first (see the root README for the exact command).

## Structure

```
src/
  app/
    layout.tsx        fonts (DM Sans, JetBrains Mono) + globals
    globals.css       design tokens (dark/light) + keyframes
    page.tsx          screen state machine (idle / loading / done / failed)
  components/         IdleHero, LoadingScreen, ResultsScreen, AnalysisFailed, ListingMap,
                      PriceTrendChart, MacroOverlay, ComparisonPanel, ScenarioModal, AppHeader, ...
  lib/
    api.ts            fetch client (same-origin proxy; failures reported, never faked)
    types.ts          backend response types
    labels.ts         SHAP feature name → friendly label
    listingSource.ts  Bayut / Dubizzle URL validation
    demo.ts           loading-step labels + scenario-modal constants
    useTheme.ts       dark/light toggle
```
