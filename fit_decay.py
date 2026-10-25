"""Fit how attention decays after a Wikipedia spike, for predictors/decay_predict.py.

Uses ONLY hours before the backtest's test period (same --test-days as backtest.py), so the
decay model never learns from the hours it is scored on.

Method:
  1. Find spikes in the training hours with the backtest's own rule (find_spikes).
  2. For each spike, take views from its peak (the busiest hour within 6 h of the onset) up to
     24 h later, subtract the page's baseline (median of the 24 h before the spike), and divide
     by the excess at the peak. Every spike then starts at 1 and decays towards 0.
  3. Take the median across spikes at each hour since the peak (robust to odd pages).
  4. Fit two curves to the DECAY PHASE only (peak until the median stops falling, at most 12 h),
     in log space:
         power law     f(t) = (1 + t/tau) ** -alpha
         exponential   f(t) = exp(-t/tau)
     and keep the one with the smaller error. After that, attention rises again about a day later
     (the "daily echo": people come back at the same time the next day); that is measured and
     reported separately rather than forced into the same curve.
  5. Save models/decay_params.json (read by decay_predict.py) and out/decay_fit.png (for the slides).

Usage:
    python fit_decay.py                       # matches python backtest.py (3 test days)
    python fit_decay.py --test-days 0.5       # matches python backtest.py --test-days 0.5
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from backtest import HOUR, find_spikes

MAX_T = 30          # hours after the peak kept (long enough to see the next-day echo)


def power(t, alpha, tau):
    return (1 + t / tau) ** -alpha


def exponential(t, tau):
    return np.exp(-t / tau)


def spike_curves(df: pd.DataFrame, train_end: pd.Timestamp) -> tuple[np.ndarray, int]:
    """Normalised excess-views curves (n_spikes, MAX_T + 1), NaN where unknown."""
    first = df["hour"].min() + 24 * HOUR
    spikes = find_spikes(df, first, train_end)
    if spikes.empty:
        raise SystemExit("no spikes in the training hours; download more data or lower --test-days")
    pages = set(spikes["page"])
    sub = df[df["page"].isin(pages) & (df["hour"] <= train_end)]
    full = pd.date_range(df["hour"].min(), train_end, freq="h")
    have = set(pd.DatetimeIndex(df.loc[df["hour"] <= train_end, "hour"].unique()))
    wide = sub.pivot_table(index="hour", columns="page", values="views", aggfunc="sum").reindex(full)
    missing_hour = ~full.isin(list(have))
    wide = wide.fillna(0.0)                       # page absent in an hour we have = 0 views...
    wide.loc[missing_hour] = np.nan               # ...but a missing download hour stays unknown

    curves = []
    for r in spikes.itertuples():
        col = wide[r.page].to_numpy()
        i = full.get_loc(r.spike_hour)
        baseline = np.nanmedian(col[max(0, i - 24):i]) if i > 0 else 0.0
        window = col[i:i + 7]
        if np.all(np.isnan(window)):
            continue
        p = i + int(np.nanargmax(window))         # peak = busiest hour within 6 h of the onset
        peak_excess = col[p] - baseline
        if not peak_excess > 0:
            continue
        seg = np.full(MAX_T + 1, np.nan)
        after = col[p:p + MAX_T + 1]
        seg[:len(after)] = np.clip(after - baseline, 0, None) / peak_excess
        curves.append(seg)
    return np.array(curves), len(spikes)


def fit(curves: np.ndarray, max_fit_h: int = 12) -> dict:
    """Fit the early decay phase only; measure the next-day echo separately.

    The median curve falls steeply for a few hours, bottoms out, then rises again about 24 h after the
    peak (people come back at the same time the next day). One smooth curve can't describe both, so:
      - decay phase: from the peak to where the median stops falling (at most max_fit_h hours)
      - daily echo:  the highest point 18-30 h after the peak, compared with the trough before it
    """
    t = np.arange(MAX_T + 1, dtype=float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # hours with no spike data yet
        med = np.nanmedian(curves, axis=0)
    n_per_t = np.sum(~np.isnan(curves), axis=0)
    valid = (med > 0) & (n_per_t >= 5)

    # End of the decay phase: first hour (after 3 h) where the median stops falling.
    fit_end = 0
    for k in range(1, min(max_fit_h, MAX_T) + 1):
        if not valid[k]:
            break
        if k > 3 and med[k] >= med[k - 1]:
            break
        fit_end = k
    if fit_end < 3:
        raise SystemExit("not enough spike data to fit the decay phase yet")
    win = valid & (t <= fit_end)
    tt, yy = t[win], np.log(med[win])

    res = {}
    (a, tau_p), _ = curve_fit(lambda x, a, tau: np.log(power(x, a, tau)), tt, yy, p0=[1.0, 1.0],
                              bounds=([0.05, 0.1], [10.0, 100.0]))
    res["power"] = {"alpha": float(a), "tau": float(tau_p),
                    "rmse_log": float(np.sqrt(np.mean((np.log(power(tt, a, tau_p)) - yy) ** 2)))}
    (tau_e,), _ = curve_fit(lambda x, tau: np.log(exponential(x, tau)), tt, yy, p0=[1.0], bounds=([0.05], [500.0]))
    res["exponential"] = {"tau": float(tau_e),
                          "rmse_log": float(np.sqrt(np.mean((np.log(exponential(tt, tau_e)) - yy) ** 2)))}
    best = min(res, key=lambda k: res[k]["rmse_log"])
    p = res[best]
    half_life = p["tau"] * (2 ** (1 / p["alpha"]) - 1) if best == "power" else p["tau"] * np.log(2)

    echo = {}
    late = valid & (t >= 18)
    trough_zone = valid & (t >= fit_end) & (t < 18)
    if late.any() and trough_zone.any():
        j = int(t[late][np.argmax(med[late])])
        trough = float(med[trough_zone].min())
        echo = {"hours_after_peak": j, "level": float(med[j]), "trough": trough,
                "echo_over_trough": float(med[j] / trough) if trough > 0 else None}
    return {"model": best, "alpha": p.get("alpha", 1.0), "tau": p["tau"], "half_life_h": float(half_life),
            "fit_window_h": int(fit_end), "candidates": res, "echo": echo,
            "median_curve": [None if np.isnan(x) else float(x) for x in med], "spikes_per_hour": n_per_t.tolist()}


def plot(params: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(MAX_T + 1, dtype=float)
    med = np.array([np.nan if x is None else x for x in params["median_curve"]])
    w = params["fit_window_h"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.axvspan(-0.3, w + 0.3, color="#e8eef8", zorder=0)
    ax.text(w / 2, 1.25, "decay phase (fitted)", ha="center", fontsize=9, color="#52514e")
    ok = med > 0
    ax.semilogy(t[ok], med[ok], "o", color="#0b0b0b", markersize=6, label=f"median of {params['n_spikes_used']} spikes")
    c = params["candidates"]
    tf = np.linspace(0, w, 100)
    tl = np.linspace(w, MAX_T, 100)
    for model, color, lab in [("exponential", "#2a78d6", f"exponential: tau = {c['exponential']['tau']:.2f} h"),
                              ("power", "#eb6834", f"power law: alpha = {c['power']['alpha']:.2f}, tau = {c['power']['tau']:.2f} h")]:
        f = (lambda x: exponential(x, c["exponential"]["tau"])) if model == "exponential" else \
            (lambda x: power(x, c["power"]["alpha"], c["power"]["tau"]))
        best = " (best)" if params["model"] == model else ""
        ax.semilogy(tf, f(tf), "-", color=color, linewidth=2, label=lab + best)
        ax.semilogy(tl, f(tl), ":", color=color, linewidth=1.2)
    e = params.get("echo") or {}
    if e:
        ax.annotate(f"daily echo, {e['hours_after_peak']} h after the peak",
                    xy=(e["hours_after_peak"], e["level"]), xytext=(e["hours_after_peak"] - 9, e["level"] * 6),
                    fontsize=9, color="#0b0b0b", arrowprops=dict(arrowstyle="->", color="#52514e"))
    ax.set_ylim(max(np.nanmin(med[ok]) / 4, 1e-4), 2)
    ax.set_xlabel("hours since the peak")
    ax.set_ylabel("extra views, as a fraction of the peak (log scale)")
    ax.set_title(f"After a spike, attention halves every {params['half_life_h'] * 60:.0f} minutes, then echoes a day later",
                 fontsize=11, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/views.parquet")
    ap.add_argument("--test-days", type=float, default=3, help="must match backtest.py")
    ap.add_argument("--top", type=int, default=150_000, help="keep top N pages per hour (as backtest.py)")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.top:
        df = df.sort_values(["hour", "views"], ascending=[True, False]).groupby("hour", sort=False).head(args.top)
    last = df["hour"].max()
    test_start = last - pd.Timedelta(days=args.test_days) + HOUR       # same rule as backtest.py
    train_end = test_start - HOUR
    print(f"fitting on hours up to {train_end} (test period starts {test_start}, never used here)")

    curves, n_spikes = spike_curves(df[df["hour"] <= train_end], train_end)
    params = fit(curves)
    params.update({"n_spikes_found": int(n_spikes), "n_spikes_used": int(len(curves)),
                   "fitted_up_to": str(train_end), "test_days": args.test_days})

    Path("models").mkdir(exist_ok=True)
    Path("out").mkdir(exist_ok=True)
    Path("models/decay_params.json").write_text(json.dumps(params, indent=2))
    plot(params, Path("out/decay_fit.png"))
    c = params["candidates"]
    print(f"{len(curves)} spikes used (of {n_spikes} found)")
    print(f"power law:   alpha={c['power']['alpha']:.3f}, tau={c['power']['tau']:.3f}  (log error {c['power']['rmse_log']:.3f})")
    print(f"exponential: tau={c['exponential']['tau']:.3f} h  (log error {c['exponential']['rmse_log']:.3f})")
    print(f"decay phase fitted on the first {params['fit_window_h']} h after the peak")
    print(f"best: {params['model']}, half-life of attention {params['half_life_h']:.2f} h "
          f"({params['half_life_h'] * 60:.0f} minutes)")
    e = params.get("echo") or {}
    if e:
        print(f"daily echo: {e['hours_after_peak']} h after the peak, extra attention is back up to {e['level']:.1%} "
              f"of the peak ({e['echo_over_trough']:.1f}x the trough of {e['trough']:.1%})")
    print("wrote models/decay_params.json and out/decay_fit.png")


if __name__ == "__main__":
    main()
