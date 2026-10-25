"""Transformer predictor (INTERFACES.md signature). Train first with:  python train_transformer.py

Predicts all horizons (1, 2, 3, 6 h) in one forward pass and caches them for the hour, because the
backtest asks for each horizon separately at the same `now`. Other horizons use the nearest one.
Set the environment variable TRANSFORMER_MODEL to score a different checkpoint (e.g. the MLP ablation).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tf_model import ARCHS, CONTEXT, DEVICE, HORIZONS, encode, predict_log
from .xgb_features import select_candidates

HOUR = pd.Timedelta(hours=1)
MODEL_PATH = Path(os.environ.get("TRANSFORMER_MODEL",
                                 Path(__file__).resolve().parent.parent / "models" / "transformer.pt"))
CANDIDATES = 200_000
SELECT_WINDOW = 25
_state: dict = {"model": None, "key": None}
torch.set_num_threads(max(1, os.cpu_count() or 1))


def _model():
    if _state["model"] is None:
        ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        m = ARCHS[ck["arch"]](**ck["kwargs"])
        m.load_state_dict(ck["state_dict"])
        m.to(DEVICE).eval()
        _state["model"] = m
    return _state["model"]


def window(history: pd.DataFrame, now: pd.Timestamp):
    """(pages, W) with W (n_pages, CONTEXT) for hours now-CONTEXT+1 .. now; NaN = hour never downloaded."""
    hour_col = history["hour"].to_numpy()
    start = now - (CONTEXT - 1) * HOUR
    lo = np.searchsorted(hour_col, np.datetime64(start), side="left")
    hi = np.searchsorted(hour_col, np.datetime64(now), side="right")
    recent = history.iloc[lo:hi]
    if recent.empty:
        return pd.Index([]), np.zeros((0, CONTEXT))
    codes, pages = pd.factorize(recent["page"])
    col = ((recent["hour"].to_numpy() - np.datetime64(start)) / np.timedelta64(1, "h")).astype(int)
    W = np.zeros((len(pages), CONTEXT), dtype=np.float32)
    W[codes, col] = recent["views"].to_numpy(dtype=np.float32)
    present = np.zeros(CONTEXT, bool)
    present[np.unique(col)] = True
    W[:, ~present] = np.nan
    keep = select_candidates(W[:, -SELECT_WINDOW:], CANDIDATES)
    return pd.Index(pages[keep]), W[keep]


def transformer_predict(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    key = (id(history), len(history), now)
    if _state["key"] != key:
        pages, W = window(history, now)
        P = predict_log(_model(), *encode(W, now)) if len(pages) else None
        _state.update(key=key, pages=pages, P=P)
    if _state["P"] is None:
        return pd.Series(dtype=float)
    j = min(range(len(HORIZONS)), key=lambda k: (abs(HORIZONS[k] - horizon), HORIZONS[k]))
    return pd.Series(np.expm1(_state["P"][:, j]).astype(float), index=_state["pages"])
