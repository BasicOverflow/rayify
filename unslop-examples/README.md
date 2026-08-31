# AI Slop → Unslop Examples

Side-by-side pairs drawn from this repo’s history: verbose AI-generated code vs the same logic after hand cleanup.

## Pattern → file

| # | Pattern | File |
|---|---------|------|
| 1 | Dead wait loops, silent excepts, emoji “status systems” | [01-wait-helpers-and-fallbacks.md](01-wait-helpers-and-fallbacks.md) |
| 2 | Legacy branches, double-checks, unused renames | [02-legacy-paths-and-double-checks.md](02-legacy-paths-and-double-checks.md) |
| 3 | Over-commented examples, section banners, print theater | [03-examples-print-theater.md](03-examples-print-theater.md) |
| 4 | Defensive `hasattr` soup + “helpful” batch auto-chunking | [04-inference-defensive-soup.md](04-inference-defensive-soup.md) |
| 5 | Regex JSON recovery / pretend structured-output | [05-json-parse-slop.md](05-json-parse-slop.md) |
| 6 | Scaffold TODOs, unused state, magic env | [06-scaffolds-magic-env.md](06-scaffolds-magic-env.md) |
| 7 | Overbuilt router bookkeeping that never paid off | [07-router-overbuild.md](07-router-overbuild.md) |
| 8 | KV/attention math inside a Ray actor → AttentionSpecs + VramReqs | [08-attention-specs-vram-planning.md](08-attention-specs-vram-planning.md) |
| 9 | Re-download bulk Kaggle blob per Ray task → cache + staging fan-out | [09-bulk-ingest-download-once-staging-fanout.md](09-bulk-ingest-download-once-staging-fanout.md) |
| 10 | Per-symbol MinIO PUT thrash → year/month fat parquet parts | [10-minio-write-thrash-vs-fat-parts.md](10-minio-write-thrash-vs-fat-parts.md) |
| 11 | One concurrency cap for Ray and MinIO → split compute vs I/O limits | [11-ray-vs-minio-concurrency-split.md](11-ray-vs-minio-concurrency-split.md) |
| 12 | Bombard yfinance/MP for discovery → NASDAQ gate + rate limits first | [12-prefilter-before-api-fanout.md](12-prefilter-before-api-fanout.md) |
| 13 | Plot PNG upload to MinIO then re-download → return bytes from worker | [13-ephemeral-bytes-not-minio-roundtrip.md](13-ephemeral-bytes-not-minio-roundtrip.md) |

Code is trimmed with `...` where the surrounding file was huge.
