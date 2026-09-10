# Lexis Markets tests

Run on the host (or any non-Docker runner) with `PYTHONPATH` set to `output/`.
Tests talk to the existing Ray cluster and infra via `.env` / `.env.test` — they are
not packaged as a compose service.

## Default: weeksim

Bare `pytest` runs the **weeksim** suite — live IO against the Ray cluster, test
namespace, and `lexis_markets_test` DB. Nights go through **cron enqueue +
`jobs.dispatch`** (same spine as the supervisor).

```powershell
cd output
pip install -r requirements-dev.txt
# once: copy .env.test.example → .env.test and fill overrides
$env:PYTHONPATH = (Get-Location).Path

pytest                    # == weeksim (default)
```

## Opt-in: unit

```powershell
pytest tests/unit -m unit
```

## Weeksim paradigm

One shared session:

1. Wipe test env (`lexis-markets-test`, `test/` lake prefix, `lexis_markets_test` DB)
2. **Day0** seed under frozen **2025-03-11** with a tiny universe (AAPL/MSFT/SPY + DFF/DGS10/UNRATE)
3. **Nights 1..7** — cron schedules EOD → `jobs.dispatch` / `submit_loop`
4. End-of-week Serve / export / chaos hooks on that spine

Quality flags are stamped when seed materialize writes L3, and when daily Serve writes a tip span. Liquid pool symbols must finish with ``suspicious_count == 0`` (no ITMR-style ``linear_ramp`` / ``sparse_bridge``). Synthetic anomaly cases live under ``tests/unit/test_quality.py``.

Requires real yfinance / FRED / MarketParquet / Kaggle. Equity EOD uses cron `target_date` (not wall `date.today()`).

Env (see [`.env.test.example`](../.env.test.example)):

- `RAY_NAMESPACE=lexis-markets-test`
- `MARKETS_LAKE_PREFIX=test/`
- `POSTGRES_*` → `lexis_markets_test`
- `MARKETS_TEST_PROFILE=1`
- `MARKETS_TEST_EOD_FULL_PATH=1` (MP + entity detect + gap scan on)
- `MARKETS_TEST_WEEK_START=2025-03-11`
- `MARKETS_TEST_WEEK_DAYS=7`

## Markers

| Marker | Purpose |
|--------|---------|
| `weeksim` | Default frozen-calendar live pipeline |
| `unit` | Offline domain (`pytest tests/unit -m unit`) |
| `io` | Live external APIs |
| `chaos` | Supervisor crash/recovery hooks (inside weeksim) |

Do not use `pytest-xdist` for weeksim (shared session spine).
