# Front end

An interactive, article-style walkthrough of the results. Plain HTML, CSS and JavaScript: no packages,
no build step and no internet connection needed.

## Running it

From the repository root, either:

```bash
uvicorn api:app --port 8000          # then open http://127.0.0.1:8000/
```

or, without Python dependencies:

```bash
python -m http.server 8000 --bind 127.0.0.1
# then open http://127.0.0.1:8000/demo/#start
```

Serve the repository root, not only `demo/`: the page reads `../demo.json` and files in `../out/`.
Open it via `localhost` (or HTTPS); browsers only allow the integrity check below in a secure context.

## Pages

| Page | What it shows |
|---|---|
| **Introduction** | The problem: a cache with limited room, and the two ways of choosing what to keep. |
| **Explore a day** | A replay of 22 October 2026: hourly article demand and both methods' hit rates with a 100,000-page cache. Play/pause, a time slider (arrow keys, Home, End), and the top articles each hour (6 rows, expandable to 50). |
| **Results** | Average hit rates at 100,000 pages, paired gains 1 and 6 hours ahead, and the foresight curve. |
| **Storage comparison** | The same method and planning time with 10,000 versus 100,000 pages. |

Navigation supports browser history and moves keyboard focus; navigating pauses playback. Technical
details are collapsible, and Reset returns to the start.

## Where the numbers come from

The front end never computes results; it only displays published files.

- **Demand** (article views per hour) comes from `demo.json`. It does not depend on cache size.
- **Hourly hit rates** on the Explore page are read from `out/hourly.csv` (cache size 100,000,
  horizon 1). `our_model` maps to `our_model_name` in `demo.json`.
- **Summary results** come from `run-metadata.json`, which copies the capacity profiles from
  `out/results.csv` and the paired comparisons from the `key_numbers.md` reports.

Missing, invalid or duplicate rows are shown as unavailable; the page never substitutes 10,000-page
results for missing 100,000-page ones. Per-article cache status is recorded in `demo.json` only for the
10,000-page replay, so the Explore page shows demand without per-article tags at 100,000 pages.

## Integrity check

`run-metadata.json` holds SHA-256 fingerprints of `demo.json`, `out/results.csv`, `out/hourly.csv`,
`out/charts/key_numbers.md` and `out/charts_100k/key_numbers.md` (line endings normalised to LF). If any
file is missing or has changed, the page marks the data as unverified and hides the result profiles.
After re-running the backtest, update the fingerprints and the published statistics together.

## Event context (optional)

`event-context.json` can add a sourced explanation for a spiking article, keyed by exact page title.
An entry is shown only if its `status` is `checked` and it has an HTTPS `source_url`. It never changes
the measurements.

## Interpreting the comparison

- **Standard rule** is `last_hour`, a persistence baseline; **Our model** is XGBoost.
- Percentages are cache hit rates (share of views served from the cache), not forecast accuracy.
- On Explore they refer to the selected hour; on Results they are averages over all evaluated hours.
- "6 hours ahead" means both methods commit to the cache contents 6 hours before the target hour.
  It shows the value of planning ahead, not that a specific spike was predicted 6 hours early.
- The denominator is the top 150,000 English Wikipedia pages each hour, not all Wikipedia traffic.
  Test targets run from 20 October 03:00 to 23 October 02:00 UTC (72/71/70/67 hours at horizons
  1/2/3/6).
