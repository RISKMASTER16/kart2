"""value_metrics.py -- what a better predictor is actually worth, compared with the standard rule.

A gain of "+1 point of hit rate" undersells a cache improvement, because what costs money is the
MISSES: every miss is a request the main servers must render. So this reports, for each model
against last_hour (the standard rule):

  1. Miss reduction    how many fewer requests reach the main servers (relative to the standard rule)
  2. Gap closed        share of the possible improvement (standard rule -> perfect foresight) captured
  3. Paired test       the gain hour by hour on the SAME hours: mean, 95% block-bootstrap interval,
                       and how often the model wins. Much stricter than comparing two separate intervals.
  4. Requests saved    extra page views served from cache per day, for this slice of Wikipedia
  5. Cache-size equivalent  how big the standard rule's cache must be to match the model
                       (needs --data; computed exactly from the views, no interpolation)
  6. Spikes            share of spike pages' views served in their first hours, and catch lag

Usage (after backtest.py):
    python value_metrics.py                                   # sections 1-4, 6 from out/
    python value_metrics.py --data data/views.parquet         # adds section 5 (takes ~1-2 min)
    python value_metrics.py --model xgboost --baseline last_24h_mean   # vs another method
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
BASE = "last_hour"


def block_bootstrap(d: np.ndarray, block: int = 6, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean of d, resampling blocks of consecutive hours (hours are correlated,
    so resampling single hours would make the interval too narrow)."""
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(d) / block))
    starts = rng.integers(0, max(len(d) - block + 1, 1), size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :len(d)]
    means = d[np.clip(idx, 0, len(d) - 1)].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test: chance of a win/loss split at least this lopsided if there were no real difference."""
    from math import comb
    n, k = wins + losses, max(wins, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def standard_rule_curves(path: str, top: int, test_days: float, horizons: list[int], max_n: int):
    """Mean hit rate of last_hour for EVERY cache size 1..max_n, per horizon, replicating backtest.py.
    Also returns mean total views per test hour."""
    df = pd.read_parquet(path, columns=["hour", "page", "views"])
    if top:
        df = df.sort_values(["hour", "views"], ascending=[True, False], kind="stable").groupby("hour", sort=False).head(top)
    by_hour = {h: g.set_index("page")["views"] for h, g in df.groupby("hour")}
    hours = pd.DatetimeIndex(sorted(by_hour))
    last = hours[-1]
    test_start = last - pd.Timedelta(days=test_days) + HOUR
    nows = [h for h in hours if h >= test_start - HOUR and h < last]
    curves, totals = {}, []
    for h in horizons:
        acc = []
        for now in nows:
            target = now + h * HOUR
            if target < test_start or target not in by_hour:
                continue
            latest = now if now in by_hour else hours[hours <= now].max()
            actual = by_hour[target]
            order = by_hour[latest].sort_values(ascending=False, kind="stable").index[:max_n]
            cum = np.cumsum(actual.reindex(order).fillna(0).to_numpy()) / actual.sum()
            acc.append(np.pad(cum, (0, max_n - len(cum)), mode="edge"))
            if h == horizons[0]:
                totals.append(actual.sum())
        curves[h] = np.mean(acc, axis=0)
    return curves, float(np.mean(totals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="folder with results.csv and hourly.csv from backtest.py")
    ap.add_argument("--data", default=None, help="views.parquet: enables the exact cache-size equivalent")
    ap.add_argument("--model", nargs="+", default=None, help="models to report (default: all but baselines)")
    ap.add_argument("--top", type=int, default=150_000, help="as backtest.py")
    ap.add_argument("--test-days", type=float, default=3, help="as backtest.py")
    ap.add_argument("--views-per-hour", type=float, default=None,
                    help="total views per hour for 'requests saved' (default: from --data or demo.json)")
    args = ap.parse_args()
    out = Path(args.out)

    res = pd.read_csv(out / "results.csv")
    hourly = pd.read_csv(out / "hourly.csv", parse_dates=["target_hour"])
    baselines = {BASE, "oracle", "last_24h_mean", "same_hour_yesterday"}
    models = args.model or [p for p in res["predictor"].unique() if p not in baselines]
    sizes, horizons = sorted(res["cache_size"].unique()), sorted(res["horizon"].unique())
    R = res.set_index(["predictor", "cache_size", "horizon"])

    curves, vph = None, args.views_per_hour
    if args.data:
        print("computing the standard rule's hit rate at every cache size (exact, from the views)...", flush=True)
        curves, mean_total = standard_rule_curves(args.data, args.top, args.test_days, horizons, max(sizes) * 2)
        vph = vph or mean_total
        for h in horizons:
            for n in sizes:
                if (BASE, n, h) in R.index and abs(curves[h][n - 1] - R.loc[(BASE, n, h), "hit_rate"]) > 0.002:
                    print(f"  WARNING: recomputed last_hour at N={n}, h={h} is {curves[h][n - 1]:.4f} but "
                          f"results.csv says {R.loc[(BASE, n, h), 'hit_rate']:.4f}; check --top/--test-days match")
    if vph is None and (out / "demo.json").exists():
        hrs = json.loads((out / "demo.json").read_text())["hours"]
        vph = float(np.mean([x["total_views"] for x in hrs])) if hrs else None

    rows = []
    for m in models:
        for n in sizes:
            for h in horizons:
                if (m, n, h) not in R.index or (BASE, n, h) not in R.index:
                    continue
                hm, hb = R.loc[(m, n, h), "hit_rate"], R.loc[(BASE, n, h), "hit_rate"]
                ho = R.loc[("oracle", n, h), "hit_rate"] if ("oracle", n, h) in R.index else np.nan
                a = hourly[(hourly.predictor == m) & (hourly.cache_size == n) & (hourly.horizon == h)]
                b = hourly[(hourly.predictor == BASE) & (hourly.cache_size == n) & (hourly.horizon == h)]
                d = a.merge(b, on="target_hour", suffixes=("_m", "_b")).sort_values("target_hour")
                diff = (d["hit_rate_m"] - d["hit_rate_b"]).to_numpy()
                lo, hi = block_bootstrap(diff)
                wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
                row = {
                    "model": m, "cache_size": n, "horizon": h,
                    "hit_rate": hm, "standard_rule": hb, "gain_pts": (hm - hb) * 100,
                    "miss_rate": 1 - hm, "standard_miss_rate": 1 - hb,
                    "miss_reduction_pct": (1 - (1 - hm) / (1 - hb)) * 100,
                    "gap_closed_pct": (hm - hb) / (ho - hb) * 100 if ho > hb else np.nan,
                    "paired_gain_ci_low_pts": lo * 100, "paired_gain_ci_high_pts": hi * 100,
                    "hours": len(diff), "hours_won": wins, "hours_lost": losses,
                    "sign_test_p": sign_test_p(wins, losses),
                    "spike_hit_rate": R.loc[(m, n, h), "spike_hit_rate"] if "spike_hit_rate" in R.columns else np.nan,
                    "standard_spike_hit_rate": R.loc[(BASE, n, h), "spike_hit_rate"] if "spike_hit_rate" in R.columns else np.nan,
                    "catch_lag_h": R.loc[(m, n, h), "median_catch_lag_h"],
                    "standard_catch_lag_h": R.loc[(BASE, n, h), "median_catch_lag_h"],
                }
                if vph:
                    row["extra_cache_hits_per_day"] = (hm - hb) * vph * 24
                if curves is not None:
                    c = curves[h]
                    k = int(np.searchsorted(c, hm))          # curve rises with N: first size reaching hm
                    row["standard_rule_size_needed"] = k + 1 if k < len(c) else np.nan
                    row["cache_size_saving_pct"] = (1 - n / (k + 1)) * 100 if k < len(c) else np.nan
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "value_metrics.csv", index=False)

    # ---- readable report
    lines = ["# What the model is worth vs the standard rule (last_hour)", "",
             f"Test period: the {args.test_days:g} days the models never saw. "
             + (f"Traffic in this data: {vph:,.0f} page views per hour." if vph else "")]
    for m in models:
        t = table[table.model == m]
        if t.empty:
            continue
        lines += ["", f"## {m}", "",
                  "| cache | ahead | hit rate (vs std) | fewer misses | gap closed | paired gain, 95% CI | hours won | "
                  + ("std rule needs | " if curves is not None else "") + ("extra hits/day |" if vph else ""),
                  "|---|---|---|---|---|---|---|" + ("---|" if curves is not None else "") + ("---|" if vph else "")]
        for r in t.itertuples():
            cell = (f"| {r.cache_size:,} | {r.horizon} h | {r.hit_rate:.2%} ({r.standard_rule:.2%}) | "
                    f"**{r.miss_reduction_pct:+.1f}%** | {r.gap_closed_pct:.0f}% | "
                    f"{r.gain_pts:+.2f} pts [{r.paired_gain_ci_low_pts:+.2f}, {r.paired_gain_ci_high_pts:+.2f}] | "
                    f"{r.hours_won}/{r.hours} (p={r.sign_test_p:.1g}) | ")
            if curves is not None:
                cell += (f"{r.standard_rule_size_needed:,.0f} pages (**{r.cache_size_saving_pct:.0f}%** less memory) | "
                         if not np.isnan(r.standard_rule_size_needed) else "beyond tested range | ")
            if vph:
                cell += f"{r.extra_cache_hits_per_day:,.0f} |"
            lines.append(cell)
        best = t.sort_values("miss_reduction_pct", ascending=False).iloc[0]
        sig = t[(t.paired_gain_ci_low_pts > 0)]
        lines += ["", f"- Biggest cut in misses: **{best.miss_reduction_pct:.1f}% fewer requests to the main servers** "
                      f"(cache {best.cache_size:,}, {best.horizon} h ahead).",
                  f"- Gain is statistically clear (paired 95% interval above zero) in {len(sig)} of {len(t)} settings."]
        if "spike_hit_rate" in t and t.spike_hit_rate.notna().any():
            s = t[(t.cache_size == sizes[len(sizes) // 2]) & (t.horizon == horizons[0])]
            if len(s):
                s = s.iloc[0]
                lines.append(f"- Spikes (cache {s.cache_size:,}, {s.horizon} h ahead): {s.spike_hit_rate:.1%} of spike pages' "
                             f"early views served from cache vs {s.standard_spike_hit_rate:.1%} for the standard rule; "
                             f"median catch lag {s.catch_lag_h:g} h vs {s.standard_catch_lag_h:g} h.")
    lines += ["", "How to read this:",
              "- *Fewer misses*: misses are what load the main servers, so this is the saving that matters to an operator.",
              "- *Paired gain*: model minus standard rule on the same hours; interval by block bootstrap (6-hour blocks).",
              "- *Hours won*: hours where the model beat the standard rule; p from an exact sign test.",
              "- *Std rule needs*: cache size at which the standard rule would match the model's hit rate.",
              "- *Extra hits/day*: for this slice (English Wikipedia, top 150k pages per hour), not all Wikimedia traffic."]
    report = "\n".join(lines)
    (out / "value_metrics.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {out / 'value_metrics.csv'} and {out / 'value_metrics.md'}")


if __name__ == "__main__":
    main()
