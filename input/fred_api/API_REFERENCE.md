# FRED API — endpoint quick reference

Base: `https://api.stlouisfed.org/fred/`

All requests require `api_key=<32-char key>`. Append `&file_type=json` for JSON.

---

## Version 2 (bulk)

```
GET /fred/v2/release/observations?release_id=18&api_key=...
```

Returns observations for **every series** on the release. Best for one-shot backfill.

Doc: [v2 index](https://fred.stlouisfed.org/docs/api/fred/v2/index.html)

---

## Version 1 — Categories

| Method | Path | Key params |
| --- | --- | --- |
| GET | `/category` | `category_id` (0=root) |
| GET | `/category/children` | `category_id` |
| GET | `/category/related` | `category_id` |
| GET | `/category/series` | `category_id`, `limit`, `offset` |
| GET | `/category/tags` | `category_id` |
| GET | `/category/related_tags` | `category_id`, `tag_names` |

Example — series in a category:
```
/category/series?category_id=32455&file_type=json&api_key=...
```

---

## Version 1 — Releases

| Method | Path | Key params |
| --- | --- | --- |
| GET | `/releases` | `limit` (max 1000), `offset`, `order_by`, `sort_order` |
| GET | `/releases/dates` | date filters |
| GET | `/release` | `release_id` |
| GET | `/release/dates` | `release_id` |
| GET | `/release/series` | `release_id`, `limit`, `offset` |
| GET | `/release/sources` | `release_id` |
| GET | `/release/tags` | `release_id` |
| GET | `/release/related_tags` | `release_id`, `tag_names` |
| GET | `/release/tables` | `release_id`, `element_id` |

Example — all series on H.15 rates release:
```
/release/series?release_id=18&file_type=json&api_key=...
```

Example — list all releases:
```
/releases?file_type=json&api_key=...
```

---

## Version 1 — Series

| Method | Path | Key params |
| --- | --- | --- |
| GET | `/series` | `series_id`, `realtime_start`, `realtime_end` |
| GET | `/series/observations` | `series_id`, `observation_start`, `observation_end`, `units`, `frequency`, `aggregation_method`, `output_type`, `vintage_dates`, `limit` (max 100000) |
| GET | `/series/categories` | `series_id` |
| GET | `/series/release` | `series_id` |
| GET | `/series/search` | `search_text`, `search_type`, `tag_names`, `exclude_tag_names`, `filter_variable`, `filter_value`, `limit` (max 1000) |
| GET | `/series/search/tags` | search context |
| GET | `/series/search/related_tags` | search + tags |
| GET | `/series/tags` | `series_id` |
| GET | `/series/updates` | `limit`, `offset`, `filter_value`, `start_time`, `end_time` |
| GET | `/series/vintagedates` | `series_id`, `realtime_start`, `realtime_end` |

### Observations — FRED mode (latest values)

```
/series/observations?series_id=DGS10&file_type=json&api_key=...
```

### Observations — ALFRED mode (as-of 2010-01-01)

```
/series/observations?series_id=UNRATE&realtime_start=2010-01-01&realtime_end=2010-01-01&file_type=json&api_key=...
```

### Observations — percent change from year ago

```
/series/observations?series_id=CPIAUCSL&units=pc1&file_type=json&api_key=...
```

### Observations — monthly from daily (downsample)

```
/series/observations?series_id=SP500&frequency=m&aggregation_method=eop&file_type=json&api_key=...
```

### Search

```
/series/search?search_text=treasury+yield&file_type=json&api_key=...
```

### Recently updated (incremental sync)

```
/series/updates?file_type=json&api_key=...
```

---

## Version 1 — Sources

| Method | Path | Key params |
| --- | --- | --- |
| GET | `/sources` | `limit`, `offset` |
| GET | `/source` | `source_id` |
| GET | `/source/releases` | `source_id` |

---

## Version 1 — Tags

| Method | Path | Key params |
| --- | --- | --- |
| GET | `/tags` | `tag_names`, `tag_group_id`, `search_text`, `limit` |
| GET | `/related_tags` | `tag_names` |
| GET | `/tags/series` | `tag_names`, `exclude_tag_names`, `limit` |

Example — all series tagged `cpi`:
```
/tags/series?tag_names=cpi&file_type=json&api_key=...
```

---

## Maps API (GeoFRED)

Same `api_key`. Separate docs under [FRED API Maps](https://fred.stlouisfed.org/docs/api/fred/).

| Area | Endpoints |
| --- | --- |
| Shape files | Regional boundaries |
| Series group meta | Map metadata |
| Series regional data | Geo-linked series values |
| Regional data | Regional harvest |

---

## Response formats

| `file_type` | Where available |
| --- | --- |
| `xml` | Default everywhere |
| `json` | Most v1 endpoints |
| `csv` | `series/observations` only (zipped) |
| `xlsx` | `series/observations` only |

---

## Units transforms (`units` param on observations)

| Code | Meaning |
| --- | --- |
| `lin` | Levels (default) |
| `chg` | Change |
| `ch1` | Change from year ago |
| `pch` | Percent change |
| `pc1` | Percent change from year ago |
| `pca` | Compounded annual rate of change |
| `cch` | Continuously compounded rate of change |
| `cca` | Continuously compounded annual rate of change |
| `log` | Natural log |

Formulas: [ALFRED growth formulas](https://alfred.stlouisfed.org/help#growth_formulas)

---

## Output types (ALFRED observations)

| `output_type` | Meaning |
| --- | --- |
| 1 | Observations by real-time period (default) |
| 2 | By vintage date, all observations |
| 3 | By vintage date, new and revised only |
| 4 | Initial release only |

---

## HTTP errors

| Code | Meaning |
| --- | --- |
| 400 | Bad request (missing/invalid `api_key`, bad params) |
| 404 | Not found |
| 423 | Locked |
| 429 | Rate limit exceeded |
| 500 | Server error |

Doc: [errors](https://fred.stlouisfed.org/docs/api/fred/errors.html)
