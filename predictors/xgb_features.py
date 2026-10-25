"""Features for the XGBoost predictor, shared by train_xgb.py and predictors/xgb.py so that
training and prediction are guaranteed to compute exactly the same thing.

Everything is built from a "window": a matrix of views with one row per page and one column per
hour, covering the 25 hours now-24 .. now (column -1 = now). A page absent in an hour we have = 0;
an hour we don't have at all (download gap) = NaN, which XGBoost handles natively.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
WINDOW = 25                                   # hours now-24 .. now
LAGS = [0, 1, 2, 3, 4, 5, 6, 12, 18, 23, 24]   # hours before `now`
CANDIDATES = 200_000
DECAY_PATH = Path(__file__).resolve().parent.parent / "models" / "decay_params.json"

FEATURES = ([f"lag_{k}" for k in LAGS] +
            ["diff_1", "diff_3", "diff_24", "mean_24", "max_24", "std_24", "hours_since_peak",
             "peak_over_now", "hours_present", "decay_pred", "is_spike", "hour_of_day", "day_of_week"])

_decay = None


def _decay_params() -> dict:
    global _decay
    if _decay is None:
        _decay = json.loads(DECAY_PATH.read_text()) if DECAY_PATH.exists() else {"model": "power", "alpha": 1.0, "tau": 1.0}
    return _decay


def _decay_f(t, p):
    return np.exp(-t / p["tau"]) if p["model"] == "exponential" else (1 + t / p["tau"]) ** -p["alpha"]


def window_from_history(history: pd.DataFrame, now: pd.Timestamp, candidates: int = CANDIDATES):
    """(pages, W) for the busiest `candidates` pages, W shaped (n_pages, WINDOW).
    history must be sorted by hour (backtest.py guarantees this)."""
    hour_col = history["hour"].to_numpy()
    lo = np.searchsorted(hour_col, np.datetime64(now - (WINDOW - 1) * HOUR), side="left")
    hi = np.searchsorted(hour_col, np.datetime64(now), side="right")
    recent = history.iloc[lo:hi]
    if recent.empty:
        return pd.Index([]), np.zeros((0, WINDOW))
    codes, pages = pd.factorize(recent["page"])
    hrs = recent["hour"].to_numpy()
    col = ((hrs - np.datetime64(now - (WINDOW - 1) * HOUR)) / np.timedelta64(1, "h")).astype(int)
    W = np.zeros((len(pages), WINDOW))
    np.add.at(W, (codes, col), recent["views"].to_numpy(dtype=float))
    present = np.zeros(WINDOW, bool)
    present[np.unique(col)] = True
    W[:, ~present] = np.nan                                       # whole missing hours stay unknown
    keep = select_candidates(W, candidates)
    return pd.Index(pages[keep]), W[keep]


def select_candidates(W: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n most relevant pages: score = max(latest views, 24 h mean)."""
    latest = _latest(W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # pages/hours with no data are expected
        score = np.maximum(np.nan_to_num(latest), np.nan_to_num(np.nanmean(W, axis=1)))
    if len(score) <= n:
        return np.arange(len(score))
    return np.argpartition(-score, n - 1)[:n]


def _latest(W: np.ndarray) -> np.ndarray:
    """Views in the latest hour that exists (if `now` itself is missing, use the hour before)."""
    have = ~np.all(np.isnan(W), axis=0)
    j = np.nonzero(have)[0]
    return W[:, j[-1]] if len(j) else np.full(W.shape[0], np.nan)


def make_features(W: np.ndarray, now: pd.Timestamp) -> pd.DataFrame:
    """Feature table (one row per page) from a window matrix W (n_pages, WINDOW)."""
    L = np.log1p(W)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows (download gaps) are expected
        f = {f"lag_{k}": L[:, -1 - k] for k in LAGS}
        f["diff_1"] = L[:, -1] - L[:, -2]
        f["diff_3"] = L[:, -1] - L[:, -4]
        f["diff_24"] = L[:, -1] - L[:, 0]
        f["mean_24"] = np.log1p(np.nanmean(W[:, 1:], axis=1))
        f["max_24"] = np.log1p(np.nanmax(W[:, 1:], axis=1))
        f["std_24"] = np.nanstd(L[:, 1:], axis=1)
        filled = np.nan_to_num(W[:, 1:], nan=-1.0)
        peak = filled.argmax(axis=1)
        hours_since_peak = (W.shape[1] - 2 - peak).astype(float)
        f["hours_since_peak"] = hours_since_peak
        latest = _latest(W)
        f["peak_over_now"] = np.log1p(filled.max(axis=1).clip(min=0)) - np.log1p(np.nan_to_num(latest))
        f["hours_present"] = np.sum(np.nan_to_num(W[:, 1:]) > 0, axis=1).astype(float)
        # The decay model's forecast as a feature (same rule as predictors/decay_predict.py): pages that
        # spiked get baseline + excess * f(t+1)/f(t); every other page gets its latest views.
        p = _decay_params()
        base = np.nan_to_num(np.nanmedian(W[:, 1:], axis=1))
        last = np.nan_to_num(latest)
        excess = np.clip(last - base, 0, None)
        peak_val = filled.max(axis=1).clip(min=0)
        spiked = (peak_val >= 3.0 * base) & (peak_val - base >= 100.0)
        decayed = base + excess * _decay_f(hours_since_peak + 1, p) / _decay_f(hours_since_peak, p)
        f["decay_pred"] = np.log1p(np.clip(np.where(spiked, decayed, last), 0, None))
        f["is_spike"] = spiked.astype(float)
    f["hour_of_day"] = np.full(W.shape[0], now.hour, dtype=float)
    f["day_of_week"] = np.full(W.shape[0], now.dayofweek, dtype=float)
    return pd.DataFrame(f)[FEATURES]
