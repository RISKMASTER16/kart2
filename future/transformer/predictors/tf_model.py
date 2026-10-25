"""Transformer forecaster shared by train_transformer.py and predictors/transformer.py, so training
and prediction encode pages in exactly the same way.

Input for one page at hour `now`: its views over the last CONTEXT hours (now-47 .. now).
  - log(1 + views), minus the page's current level (log views in the latest hour we have), so the
    model learns *shapes* (rising, fading, daily rhythm) that apply to big and small pages alike;
  - a mask for hours we never downloaded (whole-hour gaps), which the model sees explicitly;
  - hour of day (sin/cos) of every position, so it can learn the daily rhythm.
Hours are grouped into patches of PATCH hours (one token each, as in PatchTST), which cuts the
sequence from 48 to 12 tokens: cheap enough to run on a laptop CPU for 200,000 pages an hour.
A summary token carries the page's level, 24 h mean, volatility and the clock (hour of day and
weekday of the target), and its output predicts log views at now + 1, 2, 3 and 6 hours.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

HOUR = pd.Timedelta(hours=1)
CONTEXT = 48
PATCH = 4
HORIZONS = [1, 2, 3, 6]
N_GLOBAL = 9


def latest_present(L: np.ndarray) -> np.ndarray:
    """Per row, the value in the last column that is not a whole missing hour (NaN column)."""
    have = ~np.all(np.isnan(L), axis=0)
    j = np.nonzero(have)[0]
    return L[:, j[-1]] if len(j) else np.zeros(L.shape[0])


def encode(W: np.ndarray, now) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W: (n_pages, CONTEXT) raw views, NaN for hours never downloaded.
    now: one pd.Timestamp for all rows, or an array of timestamps (one per row, used in training).
    Returns tokens (n, CONTEXT//PATCH, PATCH*4), globals (n, N_GLOBAL), level (n,)  as float32."""
    n = W.shape[0]
    L = np.log1p(W)
    if isinstance(now, pd.Timestamp):
        level = np.nan_to_num(latest_present(L))
        now_hour, now_dow = np.full(n, now.hour), np.full(n, (now + HOUR).dayofweek)
        t1_hour = np.full(n, (now + HOUR).hour)
    else:   # per-row clocks; in training every sampled hour exists, so the last column is present
        now = pd.DatetimeIndex(now)
        level = np.nan_to_num(L[:, -1])
        t1 = now + HOUR
        now_hour, now_dow, t1_hour = now.hour.to_numpy(), t1.dayofweek.to_numpy(), t1.hour.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-missing rows are expected
        mean24 = np.nan_to_num(np.nanmean(L[:, -24:], axis=1))
        std24 = np.nan_to_num(np.nanstd(L[:, -24:], axis=1))
        max48 = np.nan_to_num(np.nanmax(L, axis=1))
    miss = np.isnan(L)
    x = np.where(miss, 0.0, L - level[:, None])
    hod = (now_hour[:, None] - (CONTEXT - 1) + np.arange(CONTEXT)[None, :]) % 24
    ang = 2 * math.pi * hod / 24
    per_hour = np.stack([x, miss.astype(float), np.sin(ang), np.cos(ang)], axis=-1)   # (n, CONTEXT, 4)
    tokens = per_hour.reshape(n, CONTEXT // PATCH, PATCH * 4)
    a_h, a_d = 2 * math.pi * t1_hour / 24, 2 * math.pi * now_dow / 7
    g = np.stack([level / 10, mean24 / 10, std24, (max48 - level) / 5, mean24 - level,
                  np.sin(a_h), np.cos(a_h), np.sin(a_d), np.cos(a_d)], axis=1)
    return tokens.astype(np.float32), g.astype(np.float32), level.astype(np.float32)


class PatchTransformer(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_tok = CONTEXT // PATCH
        self.patch = nn.Linear(PATCH * 4, d)
        self.glob = nn.Sequential(nn.Linear(N_GLOBAL, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(HORIZONS))
        # Skip connection from the summary features: the model starts close to "same as now"
        # and the transformer only has to learn corrections.
        self.lin = nn.Linear(N_GLOBAL, len(HORIZONS))

    def forward(self, tokens: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns predicted log(1 + views) minus the current level, for each of HORIZONS."""
        z = torch.cat([self.glob(g)[:, None], self.patch(tokens)], dim=1) + self.pos
        z = self.norm(self.enc(z))
        return self.head(z[:, 0]) + self.lin(g)


class MLPBaseline(nn.Module):
    """Same inputs, no attention: tells us whether the transformer part actually earns its keep."""
    def __init__(self, d: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        n_in = (CONTEXT // PATCH) * PATCH * 4 + N_GLOBAL
        self.net = nn.Sequential(nn.Linear(n_in, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(HORIZONS)))

    def forward(self, tokens, g):
        return self.net(torch.cat([tokens.flatten(1), g], dim=1))


ARCHS = {"transformer": PatchTransformer, "mlp": MLPBaseline}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_log(model: nn.Module, tokens: np.ndarray, g: np.ndarray, level: np.ndarray,
                batch: int = 16384) -> np.ndarray:
    """(n, len(HORIZONS)) predicted log(1 + views)."""
    model.eval()
    dev = next(model.parameters()).device
    out = np.empty((len(level), len(HORIZONS)), dtype=np.float32)
    for s in range(0, len(level), batch):
        out[s:s + batch] = model(torch.from_numpy(tokens[s:s + batch]).to(dev),
                                 torch.from_numpy(g[s:s + batch]).to(dev)).cpu().numpy()
    return np.clip(out + level[:, None], 0, None)
