"""Baseline predictors. Each follows the INTERFACES.md signature:

    predict(history, now, horizon=1) -> pd.Series of predicted views for hour now+horizon, indexed by page

history only contains rows with hour <= now, sorted by hour (backtest.py guarantees the sorting),
so rows for any hour can be found by binary search instead of scanning the whole table.
Missing hours (download gaps) fall back to the latest hour that exists.

Predictors whose output doesn't depend on `horizon` are marked `horizon_independent = True`,
so backtest.py computes them once per hour instead of once per horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def _slice(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows with start <= hour <= end, by binary search on the sorted hour column."""
    hours = history["hour"].to_numpy()
    lo = np.searchsorted(hours, np.datetime64(start), side="left")
    hi = np.searchsorted(hours, np.datetime64(end), side="right")
    return history.iloc[lo:hi]


def _views_at(history: pd.DataFrame, hour: pd.Timestamp) -> pd.Series:
    rows = _slice(history, hour, hour)
    return pd.Series(rows["views"].to_numpy(dtype=float), index=rows["page"].to_numpy())


def _latest_hour(history: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
    hours = history["hour"].to_numpy()
    i = np.searchsorted(hours, np.datetime64(now), side="right")
    return pd.Timestamp(hours[i - 1]) if i else now


def last_hour(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    """Whatever was popular in the latest hour stays popular (~ the standard 'least recently used' rule)."""
    return _views_at(history, _latest_hour(history, now))


def last_24h_mean(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    """Average views per hour over the last 24 hours (~ 'least frequently used').
    Pages absent in an hour count as 0; missing hours are left out of the average."""
    window = _slice(history, now - 23 * HOUR, now)
    n_hours = window["hour"].nunique()
    if n_hours == 0:
        return pd.Series(dtype=float)
    return window.groupby("page", sort=False)["views"].sum().astype(float) / n_hours


def same_hour_yesterday(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    """Daily rhythm: the target hour will look like the same hour 24 hours earlier.
    Falls back to last_hour if that hour is missing or not yet in the past."""
    target_yesterday = now + horizon * HOUR - 24 * HOUR
    if target_yesterday > now:
        return last_hour(history, now, horizon)
    s = _views_at(history, target_yesterday)
    return s if len(s) else last_hour(history, now, horizon)


last_hour.horizon_independent = True
last_24h_mean.horizon_independent = True
