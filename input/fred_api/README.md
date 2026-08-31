# FRED / ALFRED API — Lexis notes

Docs index: [FRED API](https://fred.stlouisfed.org/docs/api/fred/)

Free REST API from the St. Louis Fed. Returns **XML or JSON** over HTTPS. Covers **800k+** US macro / regional economic time series from 100+ government sources. Same base URLs serve both **FRED** (latest revised values) and **ALFRED** (point-in-time / vintage history).

---

## FRED vs ALFRED (read this first)

| | **FRED** | **ALFRED** |
| --- | --- | --- |
| Question answered | What is the best current value for each past date? | What was known **as of** a past date? |
| Default behavior | `realtime_start` / `realtime_end` default to **today** | Set `realtime_start` / `realtime_end` (or `vintage_dates`) to a historical window |
| Typical Lexis use | Layer 1 ingest + live-ish macro updates | ML backtests without revision lookahead; studying CPI/rates as they printed |

Same endpoints, different query params. Most Lexis work is FRED-mode. ALFRED matters when Athena/training needs **as-of-safe** macro (see [[Lexis Markets]] gap/calendar notes and Athena `series_obs` as-of design).

Refs: [FRED](https://fred.stlouisfed.org/docs/api/fred/fred.html) · [ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html) · [FRED vs ALFRED](https://fred.stlouisfed.org/docs/api/fred/fred_vs_alfred.html) · [Real-Time Periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)

### Real-time period params

- `realtime_start`, `realtime_end` — `YYYY-MM-DD`, closed interval
- Default on most endpoints: **today** (= FRED mode)
- Full history window: `realtime_start=1776-07-04&realtime_end=9999-12-31`
- ALFRED example (1980s knowledge): `realtime_start=1980-01-01&realtime_end=1989-12-31`

### ALFRED on observations (`fred/series/observations`)

Extra params beyond FRED:

| Param | Purpose |
| --- | --- |
| `vintage_dates` | Comma-separated dates; download data as it existed on those dates |
| `output_type` | `1`=by real-time period (default), `2`=by vintage all obs, `3`=new/revised only, `4`=initial release only |

Vintage query limits (from docs): e.g. output_type=2 on daily series caps at ~450 csv / 225 xlsx vintage dates; json/xml up to ~2000.

### Training datasets — revision / lookahead leakage

Default FRED pulls return **today's revised history**. Past values often get restated later (CPI, payrolls, GDP, etc.), so `(date, value)` from a normal query can include info that **didn't exist at your simulated decision time** → lookahead bias for ML/backtests.

| Use case | Mode |
| --- | --- |
| Dashboards, current research, canonical “best” history | FRED (default) |
| Features/labels at decision time *t* | ALFRED: `realtime_end=t` (or `vintage_dates` ≤ *t*) |

Also respect **release lag** — a January CPI print isn't available on Jan 1. Store `realtime_end` / vintage in Layer 1 when ingesting for training.

---

## API versions

### Version 1 — series-level (main Lexis path)

Incremental, per-series. Customize by source, release, category, tags, etc. Full endpoint list below.

Base URL: `https://api.stlouisfed.org/fred/`

### Version 2 — bulk by release

Single endpoint: **`fred/v2/release/observations`** — all series on a release, full history, in one shot. Ideal for seeding Layer 1 from whole releases (CPI, H.15 rates, etc.) instead of N per-series calls.

Ref: [API v2](https://fred.stlouisfed.org/docs/api/fred/v2/index.html) · [v2 API keys](https://fred.stlouisfed.org/docs/api/fred/v2/api_key.html)

**Auth differs from v1:** v2 requires `Authorization: Bearer YOUR_KEY` header (not `?api_key=` query param).

```python
requests.get(
    "https://api.stlouisfed.org/fred/v2/release/observations",
    params={"release_id": 18, "format": "json"},
    headers={"Authorization": f"Bearer {api_key}"},
)
```

**Strategy for Lexis:** use **v2 bulk** for initial backfill of chosen releases; use **v1** `series/observations` + `series/updates` for ongoing daily deltas.

---

## Authentication

- Free 32-char lowercase alphanumeric key via [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org)
- Query param: `api_key=...` on every request
- One key per app; each user of an app should use their own key (per docs)
- **Stored in repo** `.env` as `FRED_API_KEY` (gitignored; self-hosted Gitea). Also in Homelab `Services.md` / Lexis Markets.

```powershell
# Loaded automatically by notebooks via python-dotenv, or:
$env:FRED_API_KEY = "<from .env>"
```

Ref: [API Keys](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Rate limits & errors

| Limit | Value |
| --- | --- |
| Requests | **120 / minute / key** (documented; returns **429**) |
| Daily cap | Not in official docs; third-party guides cite ~6000/day — treat bulk v2 + caching as mandatory for large universes |

HTTP codes ([errors doc](https://fred.stlouisfed.org/docs/api/fred/errors.html)): **400**, **404**, **423 Locked**, **429**, **500**.

Practical pacing: ~0.5s between calls (~120/min). For warehouse-scale sync, prefer **FRED bulk CSV downloads** from the website over hammering the API.

---

## Common request params (v1)

| Param | Notes |
| --- | --- |
| `file_type` | `xml` (default), `json`; observations also `xlsx`, `csv` (csv returns zip) |
| `limit` / `offset` | Pagination; observations max **100,000** per request |
| `realtime_start` / `realtime_end` | ALFRED window |
| `observation_start` / `observation_end` | Filter observation dates |
| `units` | `lin`, `chg`, `ch1`, `pch`, `pc1`, `pca`, `cch`, `cca`, `log` |
| `frequency` | Downsample: `d`, `w`, `m`, `q`, `a`, … (cannot upsample) |
| `aggregation_method` | `avg`, `sum`, `eop` |

Observations missing values come back as `"."` in JSON.

---

## Full endpoint catalog (v1)

Docs root: [fred.stlouisfed.org/docs/api/fred/](https://fred.stlouisfed.org/docs/api/fred/)

### General

| Doc | Endpoint |
| --- | --- |
| [Overview](https://fred.stlouisfed.org/docs/api/fred/overview.html) | REST + HTTPS + XML/JSON |
| [FRED](https://fred.stlouisfed.org/docs/api/fred/fred.html) | What FRED data is |
| [ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html) | Vintage archive |
| [FRED vs ALFRED](https://fred.stlouisfed.org/docs/api/fred/fred_vs_alfred.html) | Default = FRED |
| [Real-Time Periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) | `realtime_*` semantics |
| [Errors](https://fred.stlouisfed.org/docs/api/fred/errors.html) | HTTP error bodies |
| [API Keys](https://fred.stlouisfed.org/docs/api/api_key.html) | Registration |

### Categories

Browse taxonomy top-down (root `category_id=0`).

| Endpoint | Purpose |
| --- | --- |
| `fred/category` | Get one category |
| `fred/category/children` | Child categories |
| `fred/category/related` | Related categories |
| `fred/category/series` | Series in category |
| `fred/category/tags` | Tags for category |
| `fred/category/related_tags` | Related tags |

Useful Lexis workflow: walk **Production & Business Activity → Prices → Consumer Price Indexes** (or search/tags) to discover CPI/PPI series for the macro universe.

### Releases

| Endpoint | Purpose |
| --- | --- |
| `fred/releases` | All releases (~158+) |
| `fred/releases/dates` | Release dates (all) |
| `fred/release` | One release metadata |
| `fred/release/dates` | Dates for one release |
| `fred/release/series` | **All series IDs on a release** ← seed series universe |
| `fred/release/sources` | Sources for release |
| `fred/release/tags` | Tags |
| `fred/release/related_tags` | Related tags |
| `fred/release/tables` | Release tables |

### Series (core data)

| Endpoint | Purpose |
| --- | --- |
| `fred/series` | Metadata (freq, units, SA, observation range, `last_updated`) |
| `fred/series/observations` | **The actual values** |
| `fred/series/categories` | Category membership |
| `fred/series/release` | Parent release |
| `fred/series/search` | Keyword search (`search_text`, tags, filters) |
| `fred/series/search/tags` | Tags for search |
| `fred/series/search/related_tags` | Related tags for search |
| `fred/series/tags` | Tags on series |
| `fred/series/updates` | Series sorted by last observation update ← **daily job hook** |
| `fred/series/vintagedates` | ALFRED revision dates |

### Sources & tags

| Endpoint | Purpose |
| --- | --- |
| `fred/sources` / `fred/source` | Data providers |
| `fred/source/releases` | Releases from source |
| `fred/tags` / `fred/related_tags` / `fred/tags/series` | Tag discovery & series lookup |

### Maps API (GeoFRED)

Regional/geo series — not all FRED series have geography.

| Endpoint | Purpose |
| --- | --- |
| Shape Files | Geo boundaries |
| Series Group Meta | Map layer metadata |
| Series Regional Data | Regional values |
| Regional Data | Regional harvest |

Same API key. Skip unless Lexis needs state/metro macro overlays.

### Version 2

| Endpoint | Purpose |
| --- | --- |
| `fred/v2/release/observations` | Bulk observations for **entire release** |

---

## Lexis Markets (`output/`)

FRED is the macro/rates source in the stitch priority chain (after Kaggle + MarketParquet + yfinance). Implementation: `output/lexis_markets/ingest.py` (`fetch_alfred_series`, Ray-batched seed).

```
Kaggle / MarketParquet / yfinance  →  equities, ETFs
FRED (ALFRED as-of)                →  macro, rates, FX indices
```

### L1 mapping

FRED is not OHLCV — value goes in `close`; OHLCV/dividend/split null. `source=fred`, `source_symbol=<FRED id>`, `realtime_start`/`realtime_end` = as-of window. Keep FRED rows separate from equity sources even when overlapping (e.g. `SP500`).

### Production seed

- `DEFAULT_FRED_SERIES` in `output/lexis_markets/config.py` (~30 core ids)
- H.15 bulk via `FRED_RELEASE_IDS = (18,)`
- Done markers under `ops/fred/done/`; reruns skip completed series

See `documentation.md` for full pipeline layout.

---

## Python access

Official: raw `requests` to `https://api.stlouisfed.org/fred/...`

Community wrapper (common): **`fredapi`** (`pip install fredapi`) — thin client over v1 endpoints.

```python
from fredapi import Fred
fred = Fred(api_key=os.environ["FRED_API_KEY"])
fred.get_series("DGS10")          # pandas Series
fred.search("treasury yield")
```

For bulk release backfill, call v2 URL directly or use `requests` + `file_type=json`.

---

## Bulk download alternative

FRED website offers full CSV/Excel dumps per series and bulk tools. For initial **Lexis Markets** warehouse seed with hundreds of series, consider:

1. Pick releases via `fred/releases` + `fred/release/series`
2. Backfill via **`fred/v2/release/observations`** (one request per release)
3. Ongoing: `series/updates` + incremental `series/observations`

Avoid 50k individual v1 calls on day one (rate limits).

---

## Files in this folder

| File | Notes |
| --- | --- |
| `README.md` | This doc |
| `API_REFERENCE.md` | Endpoint quick-reference + example URLs |
| `explore_fred_api.ipynb` | Exploration notebook (TODO) |
| `fred_series_catalog.parquet` | Cached all release series metadata (gitignored) |
| `fred_series_macro.parquet` | National macro subset after filters (gitignored) |

---

## External links

- [FRED home](https://fred.stlouisfed.org)
- [ALFRED home](https://alfred.stlouisfed.org)
- [FRED API docs](https://fred.stlouisfed.org/docs/api/fred/)
- [Register API key](https://fred.stlouisfed.org/docs/api/api_key.html)
- [Frequency aggregation FAQ](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) (in observations params)
