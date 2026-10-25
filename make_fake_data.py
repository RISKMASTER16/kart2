"""Synthetic data in exactly the INTERFACES.md formats, for development and tests without the real data.

Writes:
    data/views.parquet       hour, page, views (sections 1 and 2 of INTERFACES.md)
    data/missing_hours.txt   hours deliberately left out, to test robustness
    out/demo.json            replay day for the demo (section 5)

What the fake data contains, so baselines have something to get wrong:
    - popularity that follows a long tail (a few huge pages, many small ones)
    - a daily rhythm (busier in the evening UTC) with a slightly different peak per page
    - random noise
    - spikes: a page suddenly jumps 20-200x and fades over the next day. Some spikes build
      up for an hour or two first (catchable early); others appear from nowhere (nobody can
      predict them from hourly data).

Note: fake data has only --pages pages (default 3,000), so test the backtest with smaller cache
sizes than the real ones, e.g. 100 / 300 / 1000.

Usage:
    python make_fake_data.py
    python make_fake_data.py --pages 5000 --days 7 --seed 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORDS_A = ["Battle_of", "History_of", "List_of", "Siege_of", "University_of", "River", "Mount", "Lake"]
WORDS_B = ["Avalon", "Brightwater", "Calder", "Dunmore", "Elmstead", "Farrow", "Glenmoor", "Harlow",
           "Ivybridge", "Juniper", "Kestrel", "Larkspur", "Marlow", "Northgate", "Oakhurst", "Penrose"]


def page_names(n: int, rng: np.random.Generator) -> list[str]:
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(WORDS_A)}_{rng.choice(WORDS_B)}_{int(rng.integers(1, 10_000))}")
    return sorted(names)


def make_views(pages: int, days: int, start: str, n_spikes: int, n_missing: int,
               rng: np.random.Generator) -> tuple[pd.DataFrame, list[pd.Timestamp], list[dict]]:
    hours = pd.date_range(start, periods=days * 24, freq="h")
    T, P = len(hours), pages
    names = page_names(P, rng)

    # Long-tail popularity: rank 1 gets ~40k views/hour, rank 3000 gets ~30.
    rank = rng.permutation(P) + 1
    base = 40_000 / rank ** 0.9

    # Daily rhythm: peak around 19:00-21:00 UTC, each page slightly shifted.
    hod = hours.hour.to_numpy()[:, None]
    peak = rng.normal(20, 1.5, P)[None, :]
    rhythm = 1 + 0.45 * np.cos(2 * np.pi * (hod - peak) / 24)

    rate = base[None, :] * rhythm

    # Spikes: mostly on smaller pages (that's what makes them hard).
    spikes = []
    small_pages = np.nonzero(rank > P // 5)[0]
    for _ in range(n_spikes):
        p = int(rng.choice(small_pages))
        s = int(rng.integers(6, T - 6))
        height = float(rng.uniform(20, 200))
        build_up = int(rng.choice([0, 1, 2], p=[0.4, 0.35, 0.25]))   # hours of warning before the spike
        t = np.arange(T)
        mult = np.ones(T)
        after = t >= s
        mult[after] += height * (t[after] - s + 1.0) ** -1.2            # power-law fade
        for b in range(1, build_up + 1):                                  # early signs
            if s - b >= 0:
                mult[s - b] += height / (4 ** b)
        rate[:, p] *= mult
        spikes.append({"page": names[p], "spike_hour": hours[s], "height": height, "build_up_h": build_up})

    views = rng.poisson(rate * rng.lognormal(0, 0.15, rate.shape))

    missing = sorted(rng.choice(np.arange(24, T), size=n_missing, replace=False).tolist()) if n_missing else []
    keep_hours = np.setdiff1d(np.arange(T), missing)

    ti, pi = np.nonzero(views[keep_hours] >= 5)
    df = pd.DataFrame({
        "hour": hours[keep_hours][ti],
        "page": np.array(names)[pi],
        "views": views[keep_hours][ti, pi].astype("int64"),
    }).sort_values(["hour", "page"]).reset_index(drop=True)
    return df, [hours[m] for m in missing], spikes


def make_demo(df: pd.DataFrame, spikes: list[dict], rng: np.random.Generator, cache_frac: float = 0.1) -> dict:
    """A plausible demo.json from the fake data. Caching here is simulated (last_hour = previous hour's
    top pages; our_model = that plus some lucky extra catches), just to give the demo realistic content."""
    hours = sorted(df["hour"].unique())
    per_day = pd.Series(pd.DatetimeIndex(hours).normalize()).value_counts()
    last_full_day = per_day[per_day >= 23].index.max()          # latest day with (nearly) all 24 hours
    day_hours = [h for h in hours if pd.Timestamp(h).normalize() == last_full_day]
    by_hour = {h: g.set_index("page")["views"] for h, g in df.groupby("hour")}
    n_cache = max(1, int(df["page"].nunique() * cache_frac))

    out_hours, hr_last, hr_model = [], [], []
    for h in day_hours:
        prev = pd.Timestamp(h) - pd.Timedelta(hours=1)
        actual = by_hour[h]
        prev_top = set(by_hour[prev].nlargest(n_cache).index) if prev in by_hour else set()
        model_cache = prev_top | {p for p in actual.index if p not in prev_top and rng.random() < 0.3}
        total = int(actual.sum())
        hl = float(actual[actual.index.isin(prev_top)].sum() / total)
        hm = float(actual[actual.index.isin(model_cache)].sum() / total)
        hr_last.append(hl); hr_model.append(hm)
        top = actual.nlargest(50)
        out_hours.append({
            "hour": pd.Timestamp(h).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hit_rate": {"last_hour": round(hl, 4), "our_model": round(hm, 4)},
            "total_views": total,
            "top_pages": [{"page": p, "views": int(v),
                           "cached": {"last_hour": p in prev_top, "our_model": p in model_cache}}
                          for p, v in top.items()],
        })

    out_spikes = []
    for sp in sorted(spikes, key=lambda s: -s["height"]):
        s = sp["spike_hour"]
        if s.normalize() != last_full_day:
            continue
        window = [s + pd.Timedelta(hours=k) for k in range(-2, 7)]
        out_spikes.append({
            "page": sp["page"],
            "spike_hour": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": [w.strftime("%Y-%m-%dT%H:%M:%SZ") for w in window],
            "views": [int(by_hour[w].get(sp["page"], 0)) if w in by_hour else 0 for w in window],
            "catch_lag_h": {"last_hour": 1, "our_model": 0 if sp["build_up_h"] > 0 else 1},
        })
        if len(out_spikes) == 10:
            break

    base_l, base_m = float(np.mean(hr_last)), float(np.mean(hr_model))
    return {
        "_fake": True,
        "day": last_full_day.strftime("%Y-%m-%d"),
        "predictors": ["last_hour", "our_model"],
        "cache_size": 10000,
        "hours": out_hours,
        "spikes": out_spikes,
        "summary": {
            "hit_rate": {"last_hour": round(base_l, 4), "our_model": round(base_m, 4)},
            "foresight_curve": {
                "horizons": [1, 2, 3, 6],
                "hit_rate": {"last_hour": [round(base_l * f, 4) for f in (1.0, 0.95, 0.91, 0.83)],
                             "our_model": [round(base_m * f, 4) for f in (1.0, 0.96, 0.93, 0.87)]},
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=3000)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--start", default="2026-10-18")
    ap.add_argument("--spikes", type=int, default=40)
    ap.add_argument("--missing-hours", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df, missing, spikes = make_views(args.pages, args.days, args.start, args.spikes, args.missing_hours, rng)
    data, out = Path(args.data_dir), Path(args.out_dir)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data / "views.parquet", index=False)
    (data / "missing_hours.txt").write_text("".join(f"{m:%Y-%m-%d %H:%M:%S}\n" for m in missing))
    demo = make_demo(df, spikes, rng)
    (out / "demo.json").write_text(json.dumps(demo, indent=2))

    print(f"wrote {data / 'views.parquet'}: {len(df):,} rows, {df['page'].nunique():,} pages, "
          f"{df['hour'].nunique()} hours ({df['hour'].min()} to {df['hour'].max()})")
    print(f"wrote {data / 'missing_hours.txt'}: {len(missing)} missing hour(s)")
    print(f"wrote {out / 'demo.json'}: {len(demo['hours'])} hours, {len(demo['spikes'])} spikes on {demo['day']}")


if __name__ == "__main__":
    main()
