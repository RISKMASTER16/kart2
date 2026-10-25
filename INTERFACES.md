# Interfaces and data formats

This document specifies the contracts between the parts of the pipeline: the data file, the predictor
function, the backtest and its outputs, and the replay file read by the front end.

```
download.py ──► data/views.parquet ──► predictors/ ──► backtest.py ──► out/results.csv, out/hourly.csv
                                                                  └──► demo.json ──► demo/ (front end)
```

| Artefact | Produced by | Consumed by |
|---|---|---|
| `data/views.parquet` | `download.py`, `combine_parquets.py`, `make_fake_data.py` | everything downstream |
| predictor functions in `predictors/` | — | `backtest.py`, `quick_check.py` |
| `out/results.csv`, `out/hourly.csv` | `backtest.py` | `make_charts.py`, `value_metrics.py`, `make_xgb_report.py`, front end |
| `demo.json` | `backtest.py` (written to `out/demo.json`, published at the repository root) | front end, `api.py` |

---

## 1. Conventions

- **Time:** all timestamps are UTC. An hour is labelled by its start (`14:00` means views from 14:00 to 14:59).
- **Page names:** as in Wikipedia URLs, with underscores (`Taylor_Swift`, not `Taylor Swift`).
- **Scope:** English Wikipedia only; desktop (`en`) and mobile web (`en.m`) views are summed.
- **Excluded pages:** `Main_Page`, `-`, and titles starting with `Special:`, `File:`, `User:`, `Talk:`,
  `Wikipedia:`, `Portal:`, `Category:`, `Template:` or `Help:`.

---

## 2. `data/views.parquet`

One row per page per hour.

| Column | Type | Example | Notes |
|---|---|---|---|
| `hour` | timestamp (UTC, timezone-naive) | `2026-10-20 14:00:00` | start of the hour |
| `page` | string | `Taylor_Swift` | see conventions |
| `views` | int | `18204` | desktop + mobile |

- Rows with `views < 5` are dropped; such pages never matter for the cache.
- Exactly one row per (`hour`, `page`), sorted by `hour`, then `page`.
- Hours that could not be downloaded are absent and listed in `data/missing_hours.txt`, one per line.

```python
import pandas as pd
df = pd.read_parquet("data/views.parquet")
```

---

## 3. Predictors

Every model, from the simplest baseline to XGBoost, is one function with this signature, in its own
module in `predictors/`:

```python
def predict(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    """
    history : rows of views.parquet with hour <= now, sorted by hour.
    now     : the latest hour available.
    horizon : predict views for hour now + horizon (1 = next hour).
    returns : predicted views (float, >= 0) indexed by page name. Pages not returned
              are treated as a prediction of 0.
    """
```

- A predictor may only use rows with `hour <= now`. `backtest.py` enforces this by passing only those
  rows; predictors must not load the data file themselves.
- By default the backtest passes the last 8 days of history (`--history-days`).
- Trained models are fitted only on hours before the test period (section 4), saved to `models/`, and
  loaded once per process.
- Predictors are registered in `predictors/__init__.py`, in the `PREDICTORS` dictionary.

---

## 4. Backtest

### Settings

| Setting | Value |
|---|---|
| Test period | the last 3 days of `views.parquet`; everything earlier is training data |
| Data per hour | the top 150,000 pages by views |
| Cache sizes N | 1,000 / 10,000 / 100,000 pages |
| Horizons | 1, 2, 3 and 6 hours |
| Headline | hit rate at N = 100,000, horizon 1 |

### Procedure

For every test hour `now`, predictor, cache size N and horizon h:

1. `pred = predictor(history up to now, now, h)`
2. The cache holds the top N pages by `pred`.
3. `actual` = the real views at hour `now + h`.
4. **Hit rate** = views of cached pages in `actual` ÷ all views in `actual`.

The **oracle** ranks pages by their actual views at `now + h`; it is the ceiling, not a deployable method.

### Spikes

A page spikes at hour `s` if its views at `s` are at least 1,000 and at least 10 times its median over
the previous 24 hours (missing hours count as 0).

- **Catch lag:** first hour ≥ `s` at which the page is in the predictor's cache, minus `s`, in hours
  (0 = cached when the spike arrived; spikes not caught within 24 hours get 24). Measured at the
  replay cache size (`--demo-size`, default 10,000).
- **Spike hit rate:** share of spiking pages' views in the first 6 hours after onset served from cache.

### Outputs

`out/results.csv`: one row per predictor × cache size × horizon.

| Column | Example | Notes |
|---|---|---|
| `predictor` | `xgboost` | |
| `cache_size` | `100000` | |
| `horizon` | `1` | hours ahead |
| `hit_rate` | `0.8661` | mean over test hours |
| `hit_rate_ci_low`, `hit_rate_ci_high` | `0.8641`, `0.8682` | 95% bootstrap interval over hours |
| `median_catch_lag_h` | `0.0` | |
| `spike_hit_rate` | `0.9906` | |
| `n_hours` | `72` | test hours scored |

`out/hourly.csv`: the hit rate of every predictor × cache size × horizon for each test hour
(`predictor`, `cache_size`, `horizon`, `target_hour`, `hit_rate`). Used for paired comparisons.

```bash
python backtest.py --data data/views.parquet --out out
```

---

## 5. `demo.json` (replay file)

One replay day (the last full day of the test period), comparing `last_hour` with the best model at the
replay cache size.

```json
{
  "day": "2026-10-22",
  "predictors": ["last_hour", "our_model"],
  "our_model_name": "xgboost",
  "cache_size": 10000,
  "hours": [
    {
      "hour": "2026-10-22T14:00:00Z",
      "hit_rate": {"last_hour": 0.3941, "our_model": 0.3952},
      "total_views": 4944505,
      "top_pages": [
        {"page": "John_Anderson_(singer)", "views": 17044, "cached": {"last_hour": true, "our_model": true}}
      ]
    }
  ],
  "spikes": [
    {
      "page": "Tom_Skerritt",
      "spike_hour": "2026-10-22T20:00:00Z",
      "hours": ["2026-10-22T18:00:00Z", "2026-10-22T19:00:00Z", "2026-10-22T20:00:00Z", "..."],
      "views": [173, 439, 53683, "..."],
      "catch_lag_h": {"last_hour": 0, "our_model": 0}
    }
  ],
  "summary": {
    "hit_rate": {"last_hour": 0.4047, "our_model": 0.4059},
    "oracle_hit_rate": 0.4166,
    "foresight_curve": {
      "horizons": [1, 2, 3, 6],
      "hit_rate": {"last_hour": [0.4047, 0.3966, 0.3892, 0.374], "our_model": [0.4059, 0.399, 0.3942, 0.3867]}
    }
  }
}
```

- `top_pages`: the 50 most-viewed pages that hour, by views, descending.
- `spikes`: up to 10 of the largest spikes on the replay day, with views from 2 hours before to
  6 hours after onset (0 for hours that are missing or past the end of the data).
- Timestamps are ISO 8601 with `Z`; hit rates are fractions (0–1).
- The keys are always `last_hour` and `our_model`; `our_model_name` names the predictor behind `our_model`.

The front end verifies `demo.json`, `out/results.csv`, `out/hourly.csv` and both `key_numbers.md`
reports against SHA-256 fingerprints in `demo/run-metadata.json`. Regenerating any of them requires
updating that manifest.
