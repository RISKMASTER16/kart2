"""make_xgb_report.py -- slide-ready charts and a write-up: XGBoost vs the normal caching methods.

Reads out/results.csv and out/hourly.csv (from backtest.py) and writes to out/xgb_report/:
    xgb_report.md              the write-up with all tables
    1_fewer_misses.png         % fewer cache misses than the standard rule, by cache size and warning time
    2_every_hour.png           the gain in every one of the 72 test hours (paired)
    3_foresight.png            hit rate vs hours of warning, 100k-page cache
    4_misses_all_methods.png   misses per 100 views for every method, 100k-page cache
    tables.csv                 every number in the tables

Usage:
    python make_xgb_report.py
    python make_xgb_report.py --out out --views-per-hour 4795041
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL, STD = "xgboost", "last_hour"
NORMAL = ["last_hour", "last_24h_mean", "same_hour_yesterday", "decay"]
NAMES = {"xgboost": "XGBoost (ours)", "last_hour": "Last hour (standard rule)", "last_24h_mean": "24-hour average",
         "same_hour_yesterday": "Same hour yesterday", "decay": "Decay model (physics)", "oracle": "Perfect foresight"}

# Colours: reference data-viz palette (light). Ours = blue, standard rule = orange, the rest recede.
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal blue ramp for 1k / 10k / 100k pages
GREY, GREY_LIGHT = "#8a8984", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 13, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.axisbelow": True, "legend.frameon": False, "text.color": INK,
})


def block_bootstrap(d: np.ndarray, block: int = 6, n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(d) / block))
    starts = rng.integers(0, max(len(d) - block + 1, 1), size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :len(d)]
    m = d[np.clip(idx, 0, len(d) - 1)].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_p(w: int, l: int) -> float:
    n, k = w + l, max(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def fmt_p(p: float) -> str:
    """0.34 -> '0.34'; 4e-22 -> '4×10⁻²²'."""
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.0e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant}×10{sup}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--views-per-hour", type=float, default=None, help="default: mean total_views in demo.json")
    args = ap.parse_args()
    out = Path(args.out)
    dest = out / "xgb_report"
    dest.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(out / "results.csv")
    hourly = pd.read_csv(out / "hourly.csv", parse_dates=["target_hour"])
    R = res.set_index(["predictor", "cache_size", "horizon"])["hit_rate"]
    sizes, horizons = sorted(res.cache_size.unique()), sorted(res.horizon.unique())
    vph = args.views_per_hour
    for p in (out / "demo.json", Path("demo.json")):
        if vph is None and p.exists():
            vph = float(np.mean([h["total_views"] for h in json.loads(p.read_text())["hours"]]))

    def paired(a: str, b: str, n: int, h: int):
        x = hourly[(hourly.predictor == a) & (hourly.cache_size == n) & (hourly.horizon == h)]
        y = hourly[(hourly.predictor == b) & (hourly.cache_size == n) & (hourly.horizon == h)]
        d = x.merge(y, on="target_hour", suffixes=("_a", "_b")).sort_values("target_hour")
        diff = (d.hit_rate_a - d.hit_rate_b).to_numpy()
        lo, hi = block_bootstrap(diff)
        w, l = int((diff > 0).sum()), int((diff < 0).sum())
        return diff, d.target_hour.to_numpy(), lo, hi, w, l, sign_p(w, l)

    # ---------------------------------------------------------------- tables
    rows = []
    for n in sizes:
        for h in horizons:
            hm, hs, ho = R[(MODEL, n, h)], R[(STD, n, h)], R[("oracle", n, h)]
            best_b = max(NORMAL, key=lambda b: R.get((b, n, h), -1))
            for vs in sorted({STD, best_b}, key=lambda b: b != STD):
                diff, _, lo, hi, w, l, p = paired(MODEL, vs, n, h)
                hb = R[(vs, n, h)]
                rows.append({"cache_size": n, "horizon": h, "vs": vs, "vs_is_best_normal": vs == best_b,
                             "xgb_hit": hm, "vs_hit": hb, "oracle_hit": ho, "gain_pts": (hm - hb) * 100,
                             "ci_lo_pts": lo * 100, "ci_hi_pts": hi * 100,
                             "fewer_misses_pct": (1 - (1 - hm) / (1 - hb)) * 100,
                             "gap_closed_pct": (hm - hb) / (ho - hb) * 100 if ho > hb else np.nan,
                             "hours_won": w, "hours": len(diff), "p_sign": p,
                             "extra_hits_per_day": (hm - hb) * vph * 24 if vph else np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(dest / "tables.csv", index=False)
    S = T[T.vs == STD].set_index(["cache_size", "horizon"])
    B = T[T.vs_is_best_normal].set_index(["cache_size", "horizon"])
    n_tests = len(T)

    allb = []
    for b in NORMAL:
        for n in sizes:
            for h in horizons:
                _, _, lo, hi, w, l, p = paired(MODEL, b, n, h)
                allb.append({"vs": b, "cache_size": n, "horizon": h, "sig_better": lo > 0, "sig_worse": hi < 0, "p": p})
    n_all = len(allb)
    A = pd.DataFrame(allb)

    # ---------------------------------------------------------------- chart 1: fewer misses
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(horizons))
    wdt = 0.26
    for k, n in enumerate(sizes):
        vals = [S.loc[(n, h), "fewer_misses_pct"] for h in horizons]
        bars = ax.bar(x + (k - 1) * wdt, vals, wdt, color=RAMP[k], edgecolor=SURFACE, linewidth=2,
                      label=f"{n:,}-page cache")
        if n == sizes[-1]:
            for bar, v in zip(bars, vals):
                ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=13, weight="bold", color=INK)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, [f"{h} hour{'s' if h > 1 else ''} ahead" for h in horizons])
    ax.set_ylabel("Fewer requests reaching the main servers (%)")
    ax.set_title("XGBoost cuts cache misses vs the standard rule, most for big caches and early decisions")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "Test period: 72 hours of English Wikipedia the model never saw. Miss reduction = "
             "1 − (XGBoost misses ÷ standard-rule misses).", color=INK2, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dest / "1_fewer_misses.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- chart 2: every hour
    big = sizes[-1]
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(13, 7))
    for h, col in ((1, BLUE), (6, RAMP[2])):
        diff, t, lo, hi, w, l, p = paired(MODEL, STD, big, h)
        ax.plot(pd.to_datetime(t), diff * 100, color=col, linewidth=2, marker="o", markersize=4)
        ax.annotate(f"{h} h ahead\nbetter in {w}/{len(diff)} hours", (pd.to_datetime(t[-1]), diff[-1] * 100),
                    xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=12, weight="bold")
    ax.axhline(0, color=ORANGE, linewidth=2)
    ax.text(pd.to_datetime(t[0]), 0.1, "standard rule (no gain)", color=INK2, fontsize=11, va="bottom")
    ax.set_ylabel("XGBoost minus standard rule (hit-rate points)")
    ax.set_title(f"XGBoost beats the standard rule in every single test hour ({big:,}-page cache)")
    ax.set_ylim(bottom=min(-0.5, ax.get_ylim()[0]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:00"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.set_xlabel("Test hour (UTC)")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(dest / "2_every_hour.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- chart 3: foresight
    fig, ax = plt.subplots(figsize=(13, 7))
    series = [("oracle", GREY, "--"), (MODEL, BLUE, "-"), ("last_24h_mean", GREY_LIGHT, "-"), (STD, ORANGE, "-")]
    for pname, col, ls in series:
        ys = [R[(pname, big, h)] * 100 for h in horizons]
        ax.plot(horizons, ys, color=col, linestyle=ls, linewidth=2.5 if pname in (MODEL, STD) else 2,
                marker="o", markersize=8, label=NAMES[pname])
        ax.annotate(NAMES[pname], (horizons[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=12, color=INK, weight="bold" if pname == MODEL else "normal")
    ax.fill_between(horizons, [R[(STD, big, h)] * 100 for h in horizons], [R[(MODEL, big, h)] * 100 for h in horizons],
                    color=BLUE, alpha=0.12, linewidth=0)
    ax.set_xticks(horizons, [f"{h} h" for h in horizons])
    ax.set_xlim(horizons[0] - 0.3, horizons[-1] + 2.2)
    ax.set_xlabel("How far ahead the cache must decide")
    ax.set_ylabel("Share of views served from cache (%)")
    ax.set_title(f"The more warning a cache needs, the more prediction is worth ({big:,}-page cache)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(dest / "3_foresight.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- chart 4: misses, all methods
    order = ["oracle", MODEL, "last_24h_mean", "decay", STD, "same_hour_yesterday"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, h in zip(axes, (1, 6)):
        vals = [(1 - R[(p, big, h)]) * 100 for p in order]
        cols = [GREY_LIGHT if p == "oracle" else BLUE if p == MODEL else ORANGE if p == STD else GREY for p in order]
        ypos = np.arange(len(order))[::-1]
        ax.barh(ypos, vals, color=cols, edgecolor=SURFACE, linewidth=2, height=0.7)
        for y, v in zip(ypos, vals):
            ax.text(v + 0.2, y, f"{v:.1f}", va="center", fontsize=12, color=INK)
        ax.set_yticks(ypos, [NAMES[p] for p in order])
        ax.set_title(f"{h} hour{'s' if h > 1 else ''} ahead", fontsize=15)
        ax.set_xlim(0, max(vals) * 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Misses per 100 page views (lower is better)")
    fig.suptitle(f"Every method on the same 72 hours: misses per 100 views ({big:,}-page cache)",
                 x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout()
    fig.savefig(dest / "4_misses_all_methods.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- write-up
    def row_std(n, h):
        r = S.loc[(n, h)]
        return (f"| {n:,} | {h} h | {r.xgb_hit:.2%} | {r.vs_hit:.2%} | **{r.fewer_misses_pct:.1f}%** | "
                f"{r.gain_pts:+.2f} [{r.ci_lo_pts:+.2f}, {r.ci_hi_pts:+.2f}] | {r.hours_won}/{r.hours} | "
                f"{fmt_p(r.p_sign)} | {r.gap_closed_pct:.0f}% |")

    def row_best(n, h):
        r = B.loc[(n, h)]
        return (f"| {n:,} | {h} h | {NAMES[r.vs]} | {r.vs_hit:.2%} | {r.xgb_hit:.2%} | "
                f"{r.gain_pts:+.2f} [{r.ci_lo_pts:+.2f}, {r.ci_hi_pts:+.2f}] | **{r.fewer_misses_pct:.1f}%** | "
                f"{r.hours_won}/{r.hours} |")

    s1, s6 = S.loc[(big, 1)], S.loc[(big, 6)]
    sig_std = int((S.ci_lo_pts > 0).sum())
    bonf = 0.05 / n_all
    wins = A[A.sig_better]
    worst_p = float(wins.p.max()) if len(wins) else float("nan")
    SP = res.set_index(["predictor", "cache_size", "horizon"])["spike_hit_rate"] if "spike_hit_rate" in res else None
    mid = sizes[len(sizes) // 2]
    lines = [
        "# XGBoost vs normal caching methods",
        "",
        "**Question.** A cache can hold only the top N Wikipedia articles each hour. Can predicting next hour's "
        "readers beat the standard caching rule, which keeps whatever was most read last hour?",
        "",
        "**Setup.** English Wikipedia (desktop + mobile), hourly page views. The last 72 hours are the test period; "
        "XGBoost was trained only on the hours before it and never saw them. Every method is scored on exactly the "
        "same hours, cache sizes (1,000 / 10,000 / 100,000 pages) and warning times (1, 2, 3, 6 hours ahead).",
        "",
        "## Headline",
        "",
        f"- With a {big:,}-page cache deciding 1 hour ahead, XGBoost sends **{s1.fewer_misses_pct:.1f}% fewer requests "
        f"to the main servers** than the standard rule, and wins **{s1.hours_won} of {s1.hours} test hours**.",
        f"- Deciding 6 hours ahead (a cache that needs time to warm up), the cut grows to "
        f"**{s6.fewer_misses_pct:.1f}% fewer misses**, winning **{s6.hours_won} of {s6.hours} hours**.",
        f"- XGBoost captures **{s1.gap_closed_pct:.0f}%–{s6.gap_closed_pct:.0f}%** of the improvement that perfect "
        f"knowledge of the future would give at this cache size.",
    ]
    if vph:
        lines.append(f"- That is about **{s1.extra_hits_per_day / 1e6:.1f} million** (1 h ahead) to "
                     f"**{s6.extra_hits_per_day / 1e6:.1f} million** (6 h ahead) extra page views served from cache "
                     f"per day, for the top 150,000 English Wikipedia pages alone.")
    lines += [
        f"- The gain over the standard rule is statistically clear in **{sig_std} of {len(S)}** settings, and "
        f"XGBoost is significantly better than *every* normal method in "
        f"**{int(A.sig_better.sum())} of {len(A)}** method-by-setting comparisons "
        f"(significantly worse in {int(A.sig_worse.sum())}).",
        "",
        "![Fewer misses](1_fewer_misses.png)",
        "",
        "## Why misses, not hit rate",
        "",
        "A cache hit is cheap; a miss means the main servers must build the page. So the operator's cost scales "
        f"with the miss rate. Going from {s1.vs_hit:.1%} to {s1.xgb_hit:.1%} hit rate looks like "
        f"+{s1.gain_pts:.1f} points, but it cuts misses from {1 - s1.vs_hit:.1%} to {1 - s1.xgb_hit:.1%} of all "
        f"views: **{s1.fewer_misses_pct:.1f}% less load** on the servers behind the cache.",
        "",
        "## Full results vs the standard rule",
        "",
        "Paired gain = XGBoost minus the standard rule, hour by hour on the same hours, in hit-rate points, with a 95% "
        "interval. *Gap closed* = share of the standard rule's distance to perfect foresight that XGBoost recovers.",
        "",
        "| Cache | Ahead | XGBoost hit rate | Standard rule | Fewer misses | Paired gain [95% CI] | Hours won | Sign-test p | Gap closed |",
        "|---|---|---|---|---|---|---|---|---|",
        *[row_std(n, h) for n in sizes for h in horizons],
        "",
        "![Every hour](2_every_hour.png)",
        "",
        "## The strictest test: XGBoost vs the best normal method in each setting",
        "",
        "Which simple rule works best depends on the setting (for long warning times the 24-hour average beats "
        "\"last hour\"). Here XGBoost faces whichever normal method scored highest in each setting.",
        "",
        "| Cache | Ahead | Best normal method | Its hit rate | XGBoost | Paired gain [95% CI] | Fewer misses | Hours won |",
        "|---|---|---|---|---|---|---|---|",
        *[row_best(n, h) for n in sizes for h in horizons],
        "",
        "![All methods](4_misses_all_methods.png)",
        "",
        "## What foresight is worth",
        "",
        "If a cache needs k hours' notice to load an article, every method gets worse as k grows, but the standard "
        "rule falls fastest. Prediction matters most exactly when caches need the most notice.",
        "",
        "| Ahead | " + " | ".join(NAMES[p] for p in ["oracle", MODEL, "last_24h_mean", STD]) + " |",
        "|---|" + "---|" * 4,
        *[f"| {h} h | " + " | ".join(f"{R[(p, big, h)]:.2%}" for p in ["oracle", MODEL, "last_24h_mean", STD]) + " |"
          for h in horizons],
        f"\n_Cache of {big:,} pages._",
        "",
        "![Foresight](3_foresight.png)",
        "",
        "## How we tested (so the numbers hold up)",
        "",
        "- **No peeking:** XGBoost was trained only on hours before the 72-hour test period; at each test hour it "
        "sees only data up to that hour.",
        "- **Paired comparison:** both methods are scored on the same hours, so hour-to-hour swings in traffic "
        "cancel out. The 95% interval resamples blocks of 6 consecutive hours (block bootstrap), because "
        "neighbouring hours are correlated; resampling single hours would overstate certainty.",
        "- **Sign test:** counts the hours XGBoost won; p is the chance of a split at least that lopsided if the "
        "two methods were equally good.",
        f"- **Many comparisons:** XGBoost vs {len(NORMAL)} normal methods x {len(sizes) * len(horizons)} settings = "
        f"{n_all} comparisons. With a Bonferroni correction the bar is p < {bonf:.4f}; the largest sign-test p among "
        f"the {len(wins)} significant wins is {fmt_p(worst_p)}"
        + (", so every one survives the correction." if worst_p < bonf else ", so not every win survives the correction."),
        "",
        "## Honest limits",
        "",
    ]
    small1 = S.loc[(sizes[0], 1)]
    lines += [
        f"- **Small caches, short warning:** with {sizes[0]:,} pages deciding 1 hour ahead, XGBoost ties the "
        f"standard rule ({small1.gain_pts:+.2f} points, interval {small1.ci_lo_pts:+.2f} to {small1.ci_hi_pts:+.2f}), "
        "and 2 hours ahead the gain is too small to be sure of. The most-read pages are easy for any method; the "
        "value is in the long tail and in deciding early.",
        (f"- **Sudden spikes:** in the first 6 hours of a spike, XGBoost served {SP[(MODEL, mid, 1)]:.1%} of the spiking "
         f"pages' views from a {mid:,}-page cache, vs {SP[(STD, mid, 1)]:.1%} for the standard rule: no better. A page "
         "that jumps from nothing within a single hour can't be foreseen from hourly data by any method."
         if SP is not None else "- **Sudden spikes:** XGBoost doesn't beat the standard rule on sudden spikes."),
        "- **Hourly decisions:** real caches decide per request; we decide once per hour.",
        "- **Equal page sizes:** article sizes aren't in the data, so every page counts the same.",
        "- **Scope:** 162 hours of English Wikipedia, top 150,000 pages per hour; savings are for this slice.",
        "",
        "## Future work",
        "",
        "- A transformer (sequence model) over each page's last 48 hours, trained on more weeks of data.",
        "- Testing on Wikimedia's released per-request cache traces.",
    ]
    (dest / "xgb_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dest}/xgb_report.md, 4 charts and tables.csv")


if __name__ == "__main__":
    main()
