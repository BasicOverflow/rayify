# Jackson Crow Kaggle — dataset load notes

## Source

- **Dataset**: [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Owner**: jacksoncrow
- **Slug**: `jacksoncrow/stock-market-dataset`
- **Account used**: `basicoverflow` (Kaggle token in repo `.env` / Homelab `Services.md`)

## Contents

Historical daily prices for NASDAQ-traded stocks and ETFs (via Yahoo Finance / yfinance), coverage through ~April 2020 (~2.75 GB unzipped).

| Path | Notes |
| --- | --- |
| `stocks/` | Per-ticker CSVs (e.g. `AAPL.csv`) |
| `etfs/` | Per-ticker CSVs for ETFs |
| `symbols_valid_meta.csv` | Ticker metadata (e.g. full name) |
| `explore_jacksoncrow_kaggle.ipynb` | Schema / coverage / sample plots |

Per-file schema: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

## How to download

From repo root (`lexis-playground`). Keys in `.env`:

```powershell
pip install kaggle
Get-Content .env | ForEach-Object { if ($_ -match '^(\w+)=(.+)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }
kaggle datasets download -d jacksoncrow/stock-market-dataset -p jacksoncrow_kaggle
tar -xf jacksoncrow_kaggle\stock-market-dataset.zip -C jacksoncrow_kaggle
```

Do not use `--unzip` here: the Kaggle CLI on Windows can drop the zip path mid-extract and leave a partial `etfs/` only.

Large data files are gitignored (`*.csv`, archives). Re-download with the command above if missing.
