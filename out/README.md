# Results

Everything here is generated from the backtest on the 72 held-out test hours. Start with
[`charts_100k/key_numbers.md`](charts_100k/key_numbers.md) for the headline numbers.

| Path | Contents | Produced by |
|---|---|---|
| `results.csv` | Hit rate (with 95% interval), spike catch lag and spike hit rate for every method × cache size × horizon | `backtest.py` |
| `hourly.csv` | Hit rate for every method × cache size × horizon × test hour, for paired comparisons | `backtest.py` |
| `charts_100k/` | Main slide charts and `key_numbers.md` for the 100,000-page cache | `make_charts.py --cache-size 100000 --out out/charts_100k` |
| `charts/` | The same for the 10,000-page cache, plus a worked spike example | `make_charts.py` |
| `xgb_report/` | XGBoost against every simple method: miss reduction, paired tests, charts and write-up | `make_xgb_report.py` |
| `value_metrics.md`, `value_metrics.csv` | Miss reduction, share of the gap closed, paired significance and requests saved, per setting | `value_metrics.py` |
| `decay_fit.png` | The fitted attention-decay curve | `fit_decay.py` |

`results.csv`, `hourly.csv` and both `key_numbers.md` files are fingerprinted by the front end
(`demo/run-metadata.json`). If you regenerate them, update the fingerprints too.
