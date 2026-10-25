"""Quick sanity check of every predictor in predictors/PREDICTORS (a mini version of the backtest).

Run from the repository root, after `python make_fake_data.py` or with the real data:
    python quick_check.py
    python quick_check.py --sizes 1000 10000 100000     # real data
"""
import argparse

import numpy as np
import pandas as pd

from predictors import PREDICTORS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/views.parquet")
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000])   # small sizes for fake data
ap.add_argument("--test-hours", type=int, default=24)
args = ap.parse_args()

df = pd.read_parquet(args.data)
HOUR = pd.Timedelta(hours=1)
by_hour = {pd.Timestamp(h): g.set_index("page")["views"] for h, g in df.groupby("hour")}
hours = sorted(by_hour)
test = hours[-args.test_hours - 1:-1]          # predict each of the last N hours from the hour before

for name, predict in {**PREDICTORS, "oracle (cheats, ceiling)": None}.items():
    rates = {n: [] for n in args.sizes}
    for now in test:
        actual = by_hour.get(now + HOUR)
        if actual is None:
            continue
        pred = actual if predict is None else predict(df[df["hour"] <= now], now, 1)
        for n in args.sizes:
            cache = pred.nlargest(n).index
            rates[n].append(actual[actual.index.isin(cache)].sum() / actual.sum())
    print(f"{name:26s}" + "  ".join(f"N={n}: {np.mean(r):.3f}" for n, r in rates.items()))
