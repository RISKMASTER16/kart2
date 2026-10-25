"""Ensemble: geometric mean of the XGBoost and transformer predictions (average in log space).

Two models that are each slightly better than the standard rule usually make different mistakes,
so their average is often better than either. A page only one model predicts is averaged with 0
for the other (so it ranks lower, not higher). Needs both models trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transformer import transformer_predict
from .xgb import xgb_predict

WEIGHT_TRANSFORMER = 0.5


def ensemble_predict(history: pd.DataFrame, now: pd.Timestamp, horizon: int = 1) -> pd.Series:
    a = np.log1p(xgb_predict(history, now, horizon))
    b = np.log1p(transformer_predict(history, now, horizon))
    idx = a.index.union(b.index)
    mix = (1 - WEIGHT_TRANSFORMER) * a.reindex(idx).fillna(0) + WEIGHT_TRANSFORMER * b.reindex(idx).fillna(0)
    return np.expm1(mix)
