"""XGBoost predictor (INTERFACES.md signature). Train first with:  python train_xgb.py

Loads models/xgb_h{1,2,3,6}.json once. Features for an hour are built once and reused for every
horizon (the backtest asks for horizons 1, 2, 3 and 6 at the same `now`).
Horizons without their own model use the nearest trained one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .xgb_features import make_features, window_from_history

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_models: dict[int, xgb.XGBRegressor] = {}
_cache: dict = {"key": None}


def _model(h: int) -> xgb.XGBRegressor:
    if not _models:
        for f in sorted(MODEL_DIR.glob("xgb_h*.json")):
            m = xgb.XGBRegressor()
            m.load_model(f)
            _models[int(f.stem.split("_h")[1])] = m
        if not _models:
            raise FileNotFoundError("no models/xgb_h*.json found; run python train_xgb.py first")
    best = min(_models, key=lambda k: (abs(k - h), k))
    return _models[best]


def xgb_predict(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    key = (id(history), len(history), now)
    if _cache["key"] != key:                                 # build features once per hour
        pages, W = window_from_history(history, now)
        _cache.update(key=key, pages=pages, X=make_features(W, now) if len(pages) else None)
    if _cache["X"] is None:
        return pd.Series(dtype=float)
    pred = _model(horizon).predict(_cache["X"])
    return pd.Series(np.clip(np.expm1(pred), 0, None), index=_cache["pages"])
