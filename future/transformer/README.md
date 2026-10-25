# Future work: a transformer forecaster (experimental)

Not part of the main results. It was built and evaluated once; it is kept here as the starting
point for future work.

## What it is

A small transformer (about 74,000 parameters, small enough to train on a laptop CPU) that reads each
page's last 48 hourly views and predicts its views 1, 2, 3 and 6 hours ahead in one pass.

- Views are taken as log(1 + views) minus the page's current level, so the model learns *shapes*
  (rising, fading, daily rhythm) that apply to big and small pages alike.
- Hours are grouped into 4-hour patches (12 tokens instead of 48), as in PatchTST.
- Hours that were never downloaded are masked, and every hour carries its hour of day.
- Trained only on hours before the test period; the epoch is chosen by hit rate on the last
  12 training hours, never on the test period.

`predictors/ensemble.py` averages the transformer and XGBoost predictions in log space.

## What we found

One run, on a slightly later data window than the main results (training from 16 Oct, test period
the 72 hours ending 24 Oct), so these numbers are **not comparable** with the main README. Paired
differences over the same 72 test hours, in hit-rate points, with 95% block-bootstrap intervals:

| Cache | Ahead | Transformer vs XGBoost | XGBoost + transformer average vs XGBoost |
|---|---|---|---|
| 1,000 | 1 h | −0.01 [−0.05, +0.01] | +0.01 [−0.00, +0.02] |
| 1,000 | 6 h | +0.03 [+0.01, +0.06] | +0.05 [+0.04, +0.07] |
| 10,000 | 1 h | +0.01 [−0.01, +0.02] | +0.02 [+0.01, +0.03] |
| 10,000 | 6 h | +0.08 [+0.06, +0.11] | +0.10 [+0.07, +0.12] |
| 100,000 | 1 h | −0.05 [−0.07, −0.03] | +0.05 [+0.03, +0.06] |
| 100,000 | 6 h | +0.10 [+0.06, +0.13] | +0.17 [+0.14, +0.20] |

- On its own, the transformer roughly **matches XGBoost**: slightly better when deciding 6 hours
  ahead, slightly worse at 1 hour with a large cache.
- **Averaging the two beats XGBoost alone** in every setting (statistically clear in 5 of 6),
  though the gains are small.
- With only about 130 training hours, the model has few distinct hours to learn from. More weeks of
  data is the obvious next step.

Raw results: `out/results.csv`, `out/hourly.csv`, `out/demo.json`. Logs: `train_log.txt`,
`backtest_log.txt`.

## Running it

```bash
pip install torch
cp future/transformer/predictors/*.py predictors/
cp future/transformer/train_transformer.py .
python train_transformer.py          # -> models/transformer.pt (about 25 min on a laptop CPU)
```

Then register it in `predictors/__init__.py`:

```python
if (Path(__file__).resolve().parent.parent / "models" / "transformer.pt").exists():
    from .transformer import transformer_predict
    PREDICTORS["transformer"] = transformer_predict
    if "xgboost" in PREDICTORS:
        from .ensemble import ensemble_predict
        PREDICTORS["xgb+transformer"] = ensemble_predict
```

and score it: `python backtest.py --predictors last_hour xgboost transformer xgb+transformer --out out_tf`.
