# Jake Wright Kaggle — dataset load notes

## Source

- **Dataset**: [9000+ Tickers of Stock Market Data (Full History)](https://www.kaggle.com/datasets/jakewright/9000-tickers-of-stock-market-data-full-history)
- **Owner**: jakewright
- **Slug**: `jakewright/9000-tickers-of-stock-market-data-full-history`
- **Account used**: `basicoverflow` (Kaggle token in repo `.env` / Homelab `Services.md`)

## Files in this folder

| File | Notes |
| --- | --- |
| `jakewright_kaggle_all_stock_data.parquet` | Preferred for exploration (~1 GB) |
| `jakewright_kaggle_all_stock_data.csv` | Same data, larger (~3.5 GB) |
| `explore_jakewright_kaggle.ipynb` | Schema / coverage / sample plots |

Schema: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits` (~35M rows, ~1962–2024).

## How it was downloaded

From repo root (`lexis-playground`). Keys in `.env`:

```powershell
pip install kaggle
Get-Content .env | ForEach-Object { if ($_ -match '^(\w+)=(.+)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }
kaggle datasets download -d jakewright/9000-tickers-of-stock-market-data-full-history -p . --unzip
```

That originally produced `all_stock_data.csv` and `all_stock_data.parquet`. They were renamed and moved here:

```text
all_stock_data.*  →  jakewright_kaggle/jakewright_kaggle_all_stock_data.*
```

Large data files are gitignored (`*.parquet`, `*.csv`, archives). Re-download with the command above if missing, then rename/move into this folder.
