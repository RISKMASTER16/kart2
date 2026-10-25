"""Train the XGBoost predictor: one model per horizon (1, 2, 3, 6 hours ahead).

Uses ONLY hours before the backtest's test period (same --test-days rule as backtest.py), and only
targets that are also before it, so nothing the backtest scores is ever seen in training.

Training rows: for every --stride-th training hour `now`, the top --top-rows pages by relevance
plus a random --random-rows of the rest (so the model also sees pages that are about to appear).
Target: log(1 + views at now + h); a page absent in that hour counts as 0.
Rows are weighted by how big the page is, because the hit rate counts views: getting big pages
right matters more than getting tiny ones right.

The last 20% of training hours are held out (in time order) for early stopping, and the script
prints how much better than "same as now" the model is on them.

Usage (run fit_decay.py first, with the same --test-days, so decay_pred is a fitted feature):
    python train_xgb.py                          # matches python backtest.py
    python train_xgb.py --test-days 0.5          # matches python backtest.py --test-days 0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from predictors.xgb_features import FEATURES, WINDOW, make_features, select_candidates

HOUR = pd.Timedelta(hours=1)
HORIZONS = [1, 2, 3, 6]


def dense_matrix(df: pd.DataFrame, hours: pd.DatetimeIndex, universe: pd.Index) -> np.ndarray:
    """(n_pages, n_hours) float32 views; missing download hours are NaN, absent pages are 0."""
    sub = df[df["page"].isin(universe)]
    r = universe.get_indexer(sub["page"])
    c = hours.get_indexer(sub["hour"])
    M = np.zeros((len(universe), len(hours)), dtype=np.float32)
    M[r, c] = sub["views"].to_numpy(dtype=np.float32)
    have = hours.isin(pd.DatetimeIndex(df["hour"].unique()))
    M[:, ~have] = np.nan
    return M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/views.parquet")
    ap.add_argument("--test-days", type=float, default=3, help="must match backtest.py")
    ap.add_argument("--top", type=int, default=150_000, help="keep top N pages per hour (as backtest.py)")
    ap.add_argument("--universe", type=int, default=60_000, help="pages ever in this hourly top are eligible")
    ap.add_argument("--top-rows", type=int, default=30_000, help="most relevant pages per training hour")
    ap.add_argument("--random-rows", type=int, default=10_000, help="random other pages per training hour")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth training hour")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    df = pd.read_parquet(args.data)
    if args.top:
        df = df.sort_values(["hour", "views"], ascending=[True, False]).groupby("hour", sort=False).head(args.top)
    last = df["hour"].max()
    test_start = last - pd.Timedelta(days=args.test_days) + HOUR          # same rule as backtest.py
    train_end = test_start - HOUR
    df = df[df["hour"] <= train_end]
    hours = pd.date_range(df["hour"].min(), train_end, freq="h")
    print(f"training on hours {hours[0]} .. {train_end} (test period starts {test_start}, never used)")

    # Universe: pages that were ever in some hour's top --universe (the cache-relevant ones).
    ranked = df.sort_values(["hour", "views"], ascending=[True, False]).groupby("hour", sort=False).head(args.universe)
    universe = pd.Index(ranked["page"].unique())
    M = dense_matrix(df, hours, universe)
    print(f"{len(universe):,} pages x {len(hours)} hours loaded in {time.time() - t0:.0f}s")

    nows = list(range(WINDOW - 1, len(hours) - 1, args.stride))
    if len(nows) < 2:
        raise SystemExit("not enough training hours (need 26+ before the test period); "
                         "download more data or lower --test-days")
    Path("models").mkdir(exist_ok=True)
    report = {"train_end": str(train_end), "test_days": args.test_days, "horizons": {}}

    have = ~np.all(np.isnan(M), axis=0)                    # hours we actually downloaded
    n_skipped = sum(1 for i in nows if not (have[i] and have[i - WINDOW + 1:i + 1].mean() >= 0.75))
    if n_skipped:
        print(f"skipping {n_skipped} training hours that are missing or sit in a download gap "
              f"(need the hour itself and >= 75% of the 24 h before it)")
    for h in HORIZONS:
        usable = [i for i in nows if i + h < len(hours) and have[i + h] and have[i]
                  and have[i - WINDOW + 1:i + 1].mean() >= 0.75]
        if len(usable) < 2:
            print(f"h={h}: skipped (not enough training hours with a known target {h} h later); "
                  f"the predictor will use the nearest trained horizon")
            continue
        split = usable[max(1, int(len(usable) * 0.8))]                     # time-ordered validation
        Xs, ys, ws, is_val = [], [], [], []
        for i in usable:
            W = M[:, i - WINDOW + 1:i + 1]
            top = select_candidates(W, args.top_rows)
            rest = np.setdiff1d(np.nonzero(np.nan_to_num(W).sum(axis=1) > 0)[0], top)
            extra = rng.choice(rest, size=min(args.random_rows, len(rest)), replace=False) if len(rest) else rest
            idx = np.concatenate([top, extra])
            X = make_features(W[idx], hours[i])
            y = np.log1p(np.nan_to_num(M[idx, i + h]))
            Xs.append(X); ys.append(y)
            ws.append(1.0 + np.log1p(np.nan_to_num(W[idx, -1])))
            is_val.append(np.full(len(idx), i >= split))
        X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys)
        w = np.concatenate(ws); val = np.concatenate(is_val)
        model = xgb.XGBRegressor(n_estimators=args.n_estimators, max_depth=8, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=10, tree_method="hist",
                                 early_stopping_rounds=40, eval_metric="rmse", random_state=args.seed)
        model.fit(X[~val], y[~val], sample_weight=w[~val], eval_set=[(X[val], y[val])],
                  sample_weight_eval_set=[w[val]], verbose=False)
        pred = model.predict(X[val])
        naive = X.loc[val, "lag_0"].fillna(0).to_numpy()                    # "same as now"
        rmse = float(np.sqrt(np.average((pred - y[val]) ** 2, weights=w[val])))
        rmse_naive = float(np.sqrt(np.average((naive - y[val]) ** 2, weights=w[val])))
        model.save_model(f"models/xgb_h{h}.json")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        report["horizons"][h] = {"rows": int(len(y)), "best_iteration": int(model.best_iteration),
                                 "val_rmse_log": rmse, "val_rmse_log_naive": rmse_naive,
                                 "top_features": imp.head(6).round(3).to_dict()}
        print(f"h={h}: {len(y):,} rows, {model.best_iteration + 1} trees; validation error (log views) "
              f"{rmse:.3f} vs {rmse_naive:.3f} for 'same as now' ({1 - rmse / rmse_naive:+.0%}); "
              f"top features: {', '.join(imp.head(4).index)}  [{time.time() - t0:.0f}s]", flush=True)

    Path("models/xgb_report.json").write_text(json.dumps(report, indent=2))
    trained = sorted(report["horizons"])
    print(f"saved models for horizons {trained} (models/xgb_h*.json) and models/xgb_report.json")


if __name__ == "__main__":
    main()
