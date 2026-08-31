# lexis-playground

Scratch notebooks and source exploration for Lexis Markets. Rayified production code is in `../output/`.

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # fill keys
```

Notebooks: `load_dotenv(find_dotenv())` from `python-dotenv`.

### API keys (`.env` — gitignored)

| Variable | Service |
| --- | --- |
| `FRED_API_KEY` | St. Louis Fed FRED / ALFRED |
| `KAGGLE_API_TOKEN` | Kaggle (`basicoverflow`) |

Production EOD uses MarketParquet + yfinance (no Stooq key).

## Root

| Path | Notes |
| --- | --- |
| `README.md` | This file |
| `documentation.md` | **Current-state reference** for the rayified `output/` implementation |
| `requirements.txt` | Notebook deps |
| `.env.example` | Key template |
| `1_explore_ticker_universe.ipynb` | Ticker universe, Crow∥Wright gaps, yfinance fill experiments |
| `2_unified_schema.ipynb` | L1 obs + L2 meta/aliases POC |
| `3_stitch_l3_quality_fred.ipynb` | Stitch/quality/identity demos (POC only — not all tables made it to production) |

## `fred_api/`

| Path | Notes |
| --- | --- |
| `README.md` | FRED vs ALFRED, auth, Lexis mapping |
| `API_REFERENCE.md` | v1/v2 URL quick reference |
| `explore_fred_api.ipynb` | Catalog walk, macro filter, plots |

## `jakewright_kaggle/` · `jacksoncrow_kaggle/`

Kaggle dataset READMEs + explore notebooks. Bulk data stays local (gitignored).

## Local-only (not in git)

`poc/` parquet caches from notebooks, `.env`, Kaggle archives, `fred_*.parquet` catalog dumps.
