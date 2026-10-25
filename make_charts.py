"""Slide-ready charts and a key-numbers summary from the backtest outputs.

Reads (run backtest.py first; fit_decay.py for the decay chart):
    out/results.csv, out/demo.json, models/decay_params.json, out/decay_fit.png
Writes to out/charts/:
    1_vs_standard_rule.png difference from the standard rule on the SAME hours (paired), 1 h and 6 h ahead
    2_gap_recovered.png    share of the gap to perfect foresight each model recovers, per cache size
    3_foresight.png        advantage over the standard rule as you predict further ahead
    4_spikes.png           share of spike pages' views (first 6 h of each spike) served from cache
    5_decay_fit.png        copy of fit_decay.py's plot
    6_hit_rate_absolute.png  absolute hit rates with per-method intervals (appendix)
    7_spike_example.png    one real spike and when each method had it cached (appendix)
    key_numbers.md         every number the pitch needs, in sentences

Paired intervals: both methods are scored on the same hours, so we bootstrap the per-hour DIFFERENCE,
resampling blocks of 6 consecutive hours (neighbouring hours are correlated, so resampling single
hours would make the intervals too narrow).

Usage:
    python make_charts.py
    python make_charts.py --cache-size 100000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colours (validated palette): the two models get the colour; baselines stay neutral grey.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"xgboost": "#2a78d6", "decay": "#eb6834"}
NEUTRAL, CEILING = "#a3a29c", "#3d3c39"
NAMES = {
    "last_hour": "Last hour (standard rule)",
    "last_24h_mean": "24-hour average",
    "same_hour_yesterday": "Same hour yesterday",
    "decay": "Physics: decay model",
    "xgboost": "XGBoost + decay feature",
    "oracle": "Perfect foresight (ceiling)",
}
MODELS = ["decay", "xgboost"]


def style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(axis="x" if ax.get_xlabel() == "" else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=24 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom")


def name(p: str) -> str:
    return NAMES.get(p, p)


def chart_hit_rate(r: pd.DataFrame, n: int, out: Path) -> None:
    d = r[(r["horizon"] == 1) & (r["cache_size"] == n)].set_index("predictor")
    oracle = d.loc["oracle", "hit_rate"] if "oracle" in d.index else None
    d = d.drop(index="oracle", errors="ignore").sort_values("hit_rate")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(d) + 2.2))
    y = np.arange(len(d))
    for i, (p, row) in enumerate(d.iterrows()):
        c = COLOR.get(p, NEUTRAL)
        lo, hi = row["hit_rate_ci_low"], row["hit_rate_ci_high"]
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([lo * 100, hi * 100], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.scatter(row["hit_rate"] * 100, i, s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        left = (lo if pd.notna(lo) else row["hit_rate"]) * 100      # label left of the interval, clear of the ceiling line
        ax.text(left, i, f"{row['hit_rate']:.2%}  ", ha="right", va="center", fontsize=9, color=INK)
    if oracle is not None:
        ax.axvline(oracle * 100, color=CEILING, linestyle="--", linewidth=1.5)
        ax.text(oracle * 100, -0.75, f" perfect foresight {oracle:.2%}", fontsize=9, color=INK_2, va="bottom")
    ax.set_yticks(y, [name(p) for p in d.index], color=INK)
    ax.set_xlabel("")
    lo_x = min(d["hit_rate_ci_low"].min(), d["hit_rate"].min()) * 100
    hi_x = (oracle if oracle is not None else d["hit_rate"].max()) * 100
    pad = (hi_x - lo_x) * 0.15 + 0.1
    ax.set_xlim(lo_x - pad * 3, hi_x + pad * 2.5)
    ax.set_ylim(-0.8, len(d) - 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    style(ax, f"Hit rate with room for {n:,} pages, 1 hour ahead",
          "share of next hour's Wikipedia views served from the cache · lines = 95% intervals · axis zoomed in")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def gap_table(r: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    d = r[r["horizon"] == h].pivot(index="cache_size", columns="predictor", values="hit_rate")
    gap = d["oracle"] - d["last_hour"]
    return pd.DataFrame({m: (d[m] - d["last_hour"]) / gap for m in MODELS if m in d.columns})


def chart_gap(r: pd.DataFrame, out: Path) -> None:
    g = gap_table(r) * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(g))
    width = 0.36
    models = [m for m in MODELS if m in g.columns]
    for k, m in enumerate(models):
        xs = x + (k - (len(models) - 1) / 2) * (width + 0.02)
        bars = ax.bar(xs, g[m], width, color=COLOR[m], label=name(m), edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, g[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -1.5), f"{v:+.0f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, [f"{int(n):,} pages" for n in g.index], color=INK)
    ax.set_ylabel("% of the gap recovered", color=INK_2)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    lo, hi = min(0, g.min().min()), max(0, g.max().max())
    ax.set_ylim(lo - 0.15 * (hi - lo) - 1, hi + 0.15 * (hi - lo) + 1)
    style(ax, "How much of the possible improvement each model captures",
          "0% = standard rule (last hour) · 100% = perfect foresight · 1 hour ahead")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def chart_foresight(r: pd.DataFrame, n: int, out: Path) -> pd.DataFrame:
    d = r[r["cache_size"] == n].pivot(index="horizon", columns="predictor", values="hit_rate").sort_index()
    adv = d.sub(d["last_hour"], axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(adv))
    if "oracle" in adv.columns:
        ax.plot(x, adv["oracle"], linestyle="--", color=CEILING, linewidth=1.5)
        ax.text(x[-1] + 0.08, adv["oracle"].iloc[-1], "perfect foresight", fontsize=9, color=INK_2, va="center")
    for m in MODELS:
        if m in adv.columns:
            ax.plot(x, adv[m], color=COLOR[m], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(x[-1] + 0.08, adv[m].iloc[-1], name(m), fontsize=9, color=INK, va="center")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.text(-0.15, 0, "standard rule ", fontsize=9, color=INK_2, va="top")
    ax.set_xticks(x, [f"{h} h" for h in adv.index], color=INK)
    ax.set_xlim(-0.4, len(x) - 1 + 1.3)
    ax.set_xlabel("how far ahead the cache is decided", color=INK_2)
    ax.set_ylabel("hit rate above the standard rule (points)", color=INK_2)
    style(ax, "What is foresight worth?", f"room for {n:,} pages · advantage over 'last hour' as the horizon grows")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return adv


def chart_spike(demo: dict, out: Path) -> dict | None:
    spikes = demo.get("spikes") or []
    if not spikes:
        return None
    better = [s for s in spikes if s["catch_lag_h"].get("our_model", 99) < s["catch_lag_h"].get("last_hour", 99)]
    s = max(better or spikes, key=lambda s: max(s["views"]))
    hours = pd.to_datetime(s["hours"])
    views = np.array(s["views"])
    spike_i = list(s["hours"]).index(s["spike_hour"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(views))
    ax.plot(x, views, color=INK_2, linewidth=2, marker="o", markersize=6, markerfacecolor=INK_2,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    model_name = demo.get("our_model_name", "our_model")
    labels = {"last_hour": ("Last hour", NEUTRAL), "our_model": (name(model_name), COLOR.get(model_name, COLOR["xgboost"]))}
    top = views.max()
    captions = []
    for k, (key, (lab, c)) in enumerate(labels.items()):
        lag = s["catch_lag_h"].get(key)
        if lag is None:
            continue
        xi = spike_i + lag + (-0.07 if key == "last_hour" else 0.07)
        if xi < len(x):
            ax.axvline(xi, color=c, linewidth=2.5, linestyle="-" if key == "our_model" else (0, (1, 1.5)))
        captions.append((f"{lab}: in cache {lag} h after the spike", c))
    for k, (txt, c) in enumerate(captions):
        ax.text(0.97, 0.95 - 0.08 * k, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK)
        ax.plot([0.985, 0.985], [0.95 - 0.08 * k - 0.05, 0.95 - 0.08 * k], transform=ax.transAxes, color=c,
                linewidth=2.5, linestyle="-" if k else (0, (1, 1.5)), clip_on=False)
    ax.set_xticks(x, [f"{h:%H:%M}" for h in hours], color=INK, fontsize=9)
    ax.set_ylabel("views per hour", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    style(ax, f"Spike: {s['page'].replace('_', ' ')}", f"{hours[0]:%d %b} · views per hour (UTC) and when each cache had it")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return s


def block_bootstrap_ci(d: np.ndarray, block: int = 6, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for mean(d), resampling blocks of consecutive hours (moving block bootstrap)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(hourly: pd.DataFrame, n: int, horizons=(1, 6)) -> pd.DataFrame:
    """Mean per-hour difference from last_hour (in points) with block-bootstrap 95% intervals."""
    rows = []
    for h in horizons:
        w = hourly[(hourly["cache_size"] == n) & (hourly["horizon"] == h)].pivot(
            index="target_hour", columns="predictor", values="hit_rate").sort_index()
        if "last_hour" not in w.columns:
            continue
        for p in w.columns:
            if p == "last_hour":
                continue
            d = ((w[p] - w["last_hour"]) * 100).dropna().to_numpy()
            lo, hi = block_bootstrap_ci(d)
            rows.append({"predictor": p, "horizon": h, "diff_pts": float(d.mean()), "ci_low": lo, "ci_high": hi,
                         "share_hours_better": float((d > 0).mean()), "n_hours": len(d)})
    return pd.DataFrame(rows)


def chart_paired(pt: pd.DataFrame, n: int, out: Path) -> None:
    hs = sorted(pt["horizon"].unique())
    order = [p for p in ["xgboost", "decay", "last_24h_mean", "same_hour_yesterday"] if p in set(pt["predictor"])]
    fig, axes = plt.subplots(1, len(hs), figsize=(5.2 * len(hs), 0.55 * len(order) + 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    lo_all = min(pt["ci_low"].min(), pt["diff_pts"].min(), 0)
    hi_all = max(pt["ci_high"].max(), 0)
    orc = pt[pt["predictor"] == "oracle"]
    span = hi_all - lo_all
    for ax, h in zip(axes, hs):
        d = pt[pt["horizon"] == h].set_index("predictor")
        ceiling_txt = None
        for i, p in enumerate(order[::-1]):
            if p not in d.index:
                continue
            r = d.loc[p]
            c = COLOR.get(p, NEUTRAL)
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, linewidth=2.5, solid_capstyle="round")
            ax.scatter(r["diff_pts"], i, s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            sig = "" if r["ci_low"] > 0 or r["ci_high"] < 0 else " (n.s.)"
            # Values in a column just right of the plot, so they never sit on the reference lines.
            ax.text(1.02, i, f"{r['diff_pts']:+.2f}{sig}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=9.5, color=INK, clip_on=False)
        ax.axvline(0, color=INK_2, linewidth=1.2)
        if h in set(orc["horizon"]):
            o = float(orc[orc["horizon"] == h]["diff_pts"].iloc[0])
            ax.axvline(o, color=CEILING, linestyle="--", linewidth=1.2)
            ceiling_txt = f"dashed = perfect foresight (+{o:.2f})"
        ax.set_yticks(range(len(order)), [name(p) for p in order[::-1]], color=INK)
        right = max(hi_all, orc["diff_pts"].max() if len(orc) else 0)
        ax.set_xlim(lo_all - span * 0.08, right + span * 0.08)
        ax.set_ylim(-0.6, len(order) - 0.1)
        ax.set_xlabel("hit-rate points vs the standard rule (same hours)", color=INK_2)
        style(ax, f"{h} hour{'s' if h > 1 else ''} ahead",
              ceiling_txt if h in set(orc["horizon"]) else None)
    fig.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 0.97, 0.86), w_pad=5)
    fig.suptitle(f"Improvement over the standard rule, room for {n:,} pages", x=0.01, y=0.98, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "dots = average difference on the same test hours · lines = 95% intervals "
             "(block bootstrap) · n.s. = not significant", fontsize=10, color=INK_2)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def chart_spike_share(r: pd.DataFrame, n: int, out: Path) -> pd.Series | None:
    if "spike_hit_rate" not in r.columns:
        return None
    d = r[(r["horizon"] == 1) & (r["cache_size"] == n)].set_index("predictor")["spike_hit_rate"].dropna()
    if d.empty:
        return None
    d = d.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(d) + 2))
    colors = [COLOR.get(p, CEILING if p == "oracle" else NEUTRAL) for p in d.index]
    ax.barh(range(len(d)), d * 100, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(d):
        ax.text(v * 100 + 0.8, i, f"{v:.1%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(d)), [name(p) for p in d.index], color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    style(ax, "Who serves the spikes?", f"share of spike pages' views in their first 6 h served from cache · {n:,} pages")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return d



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="out/results.csv")
    ap.add_argument("--demo", default="out/demo.json")
    ap.add_argument("--decay", default="models/decay_params.json")
    ap.add_argument("--cache-size", type=int, default=10000)
    ap.add_argument("--out", default="out/charts")
    args = ap.parse_args()

    r = pd.read_csv(args.results)
    sizes = sorted(r["cache_size"].unique())
    n = args.cache_size if args.cache_size in sizes else sizes[len(sizes) // 2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for old in ["1_hit_rate.png", "4_spike_story.png"]:        # names used by earlier versions
        (out / old).unlink(missing_ok=True)
    pt = pd.DataFrame()
    hourly_path = Path(args.results).with_name("hourly.csv")
    if hourly_path.exists():
        pt = paired_table(pd.read_csv(hourly_path, parse_dates=["target_hour"]), n)
        if not pt.empty:
            chart_paired(pt, n, out / "1_vs_standard_rule.png")
    else:
        print(f"note: {hourly_path} not found (rerun backtest.py) - skipping the paired comparison")
    chart_gap(r, out / "2_gap_recovered.png")
    adv = chart_foresight(r, n, out / "3_foresight.png")
    spike_share = chart_spike_share(r, n, out / "4_spikes.png")
    if Path("out/decay_fit.png").exists():
        shutil.copy2("out/decay_fit.png", out / "5_decay_fit.png")
    chart_hit_rate(r, n, out / "6_hit_rate_absolute.png")
    demo = json.loads(Path(args.demo).read_text()) if Path(args.demo).exists() else {}
    spike = chart_spike(demo, out / "7_spike_example.png")

    # ---- key numbers, written as sentences for the slides ----
    h1 = r[(r["horizon"] == 1) & (r["cache_size"] == n)].set_index("predictor")
    gaps = gap_table(r)
    hours = int(h1["n_hours"].max())
    lines = [f"# Key numbers (test period: {hours} hours the models never saw)", ""]
    lines.append(f"## Cache with room for {n:,} pages, deciding 1 hour ahead")
    for p in h1.sort_values("hit_rate", ascending=False).index:
        row = h1.loc[p]
        ci = (f" (95% interval {row['hit_rate_ci_low']:.2%}–{row['hit_rate_ci_high']:.2%})"
              if pd.notna(row["hit_rate_ci_low"]) else "")
        lines.append(f"- {name(p)}: **{row['hit_rate']:.2%}** of views served from cache{ci}")
    if {"oracle", "last_hour"} <= set(h1.index):
        gap = h1.loc["oracle", "hit_rate"] - h1.loc["last_hour", "hit_rate"]
        lines.append(f"- The standard rule leaves **{gap * 100:.2f} points** on the table vs perfect foresight.")
    if not pt.empty:
        lines += ["", "## Same-hours comparison with the standard rule (paired, 95% block-bootstrap intervals)"]
        for _, row in pt[pt["predictor"].isin(MODELS + ["oracle"])].sort_values(["horizon", "diff_pts"],
                                                                               ascending=[True, False]).iterrows():
            sig = "significant" if (row["ci_low"] > 0 or row["ci_high"] < 0) else "NOT significant"
            lines.append(f"- {int(row['horizon'])} h ahead, {name(row['predictor'])}: **{row['diff_pts']:+.2f} points** "
                         f"(95% interval {row['ci_low']:+.2f} to {row['ci_high']:+.2f}, {sig}); "
                         f"better in {row['share_hours_better']:.0%} of hours")
    lines += ["", "## Share of that gap each model recovers (1 hour ahead)"]
    for size, row in gaps.iterrows():
        lines.append(f"- {int(size):,} pages: " + ", ".join(f"{name(m)} **{v:+.0%}**" for m, v in row.items()))
    lines += ["", f"## Foresight (room for {n:,} pages): hit-rate points above the standard rule"]
    for h, row in adv.iterrows():
        lines.append(f"- {h} h ahead: " + ", ".join(f"{name(m)} {row[m]:+.2f}" for m in MODELS + ['oracle'] if m in row))
    lag = h1["median_catch_lag_h"]
    lines += ["", "## Spikes", "- Median hours late catching a spike: " +
              ", ".join(f"{name(p)} {v:g} h" for p, v in lag.items())]
    if spike_share is not None:
        lines.append("- Share of spike pages' views (first 6 h of each spike) served from cache: " +
                     ", ".join(f"{name(p)} {v:.1%}" for p, v in spike_share.sort_values(ascending=False).items()))
    if spike:
        lags = spike["catch_lag_h"]
        lines.append(f"- Example: **{spike['page'].replace('_', ' ')}** peaked at {max(spike['views']):,} views/hour; "
                     f"the standard rule had it cached {lags.get('last_hour')} h after the spike, "
                     f"{name(demo.get('our_model_name', 'our_model'))} after {lags.get('our_model')} h.")
    if Path(args.decay).exists():
        d = json.loads(Path(args.decay).read_text())
        model = "power law" if d["model"] == "power" else "exponential decay"
        shape = f"alpha = {d['alpha']:.2f}, tau = {d['tau']:.2f} h" if d["model"] == "power" else f"tau = {d['tau']:.2f} h"
        lines += ["", "## Physics of attention",
                  f"- After a spike, extra attention fades like a **{model}** ({shape}), fitted on "
                  f"{d.get('n_spikes_used', '?')} spikes from the training days.",
                  f"- **Half-life of attention: about {d['half_life_h'] * 60:.0f} minutes** after the peak "
                  f"(decay phase fitted on the first {d.get('fit_window_h', '?')} h)."]
        e = d.get("echo") or {}
        if e:
            lines.append(f"- **Daily echo:** {e['hours_after_peak']} h after the peak, extra attention is back up to "
                         f"{e['level']:.1%} of the peak, {e['echo_over_trough']:.1f}x its low point.")
    (out / "key_numbers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\ncharts written to {out}/")


if __name__ == "__main__":
    main()
