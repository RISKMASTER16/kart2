"""Train the transformer predictor (predictors/transformer.py) on hours BEFORE the backtest's test
period, exactly like train_xgb.py: same --test-days rule, same top-150k-per-hour data.

Model choice uses the real metric, not just the loss: after every epoch the model ranks pages
for each validation hour (the last --val-hours of the training period, never the test period)
and we measure the hit rate at 1k / 10k / 100k pages against "last hour" and perfect foresight.
The epoch with the best validation hit rate is saved.

Then, for the final model, retrain on ALL training hours (validation included) for that many
epochs with --val-hours 0 --epochs N.

Usage:
    python train_transformer.py                              # train + validate, saves best epoch
    python train_transformer.py --val-hours 0 --epochs 6     # final fit on all training hours
    python train_transformer.py --arch mlp                   # same inputs, no attention (ablation)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predictors.tf_model import ARCHS, CONTEXT, DEVICE, HORIZONS, encode, predict_log
from predictors.xgb_features import select_candidates

HOUR = pd.Timedelta(hours=1)
SELECT_WINDOW = 25            # candidates chosen from the last 25 hours, as for XGBoost
CANDIDATES = 200_000


def load(path: str, top: int, test_days: float):
    df = pd.read_parquet(path, columns=["hour", "page", "views"])
    if top:
        df = df.sort_values(["hour", "views"], ascending=[True, False]).groupby("hour", sort=False).head(top)
    last = df["hour"].max()
    test_start = last - pd.Timedelta(days=test_days) + HOUR            # same rule as backtest.py
    train_end = test_start - HOUR
    df = df[df["hour"] <= train_end]
    hours = pd.date_range(df["hour"].min(), train_end, freq="h")
    codes, pages = pd.factorize(df["page"])
    M = np.zeros((len(pages), len(hours)), dtype=np.float32)
    M[codes, hours.get_indexer(df["hour"])] = df["views"].to_numpy(dtype=np.float32)
    have = hours.isin(pd.DatetimeIndex(df["hour"].unique()))
    M[:, ~have] = np.nan
    return M, hours, have, pd.Index(pages), test_start


def windows(Mp: np.ndarray, rows: np.ndarray, nows: np.ndarray) -> np.ndarray:
    """Views for rows over hours now-CONTEXT+1 .. now. Mp is M padded on the left by CONTEXT-1 NaNs,
    so column now+CONTEXT-1 of Mp is hour `now`."""
    cols = nows[:, None] + np.arange(CONTEXT)[None, :]
    return Mp[rows[:, None], cols]


def hit_rates(pred: np.ndarray, cand: np.ndarray, actual: np.ndarray, sizes) -> list[float]:
    """Same as backtest.py: share of all views at the target hour landing on the top-N predicted pages."""
    order = cand[np.argsort(-pred, kind="stable")]
    cum = np.cumsum(np.nan_to_num(actual[order]))
    total = np.nansum(actual)
    return [float(cum[min(n, len(cum)) - 1] / total) for n in sizes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/views.parquet")
    ap.add_argument("--test-days", type=float, default=3, help="must match backtest.py")
    ap.add_argument("--top", type=int, default=150_000)
    ap.add_argument("--val-hours", type=int, default=12, help="last training hours used to pick the epoch (0 = none)")
    ap.add_argument("--arch", choices=list(ARCHS), default="transformer")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--samples-per-epoch", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--top-rows", type=int, default=30_000, help="most relevant pages per training hour")
    ap.add_argument("--random-rows", type=int, default=10_000, help="random other pages per training hour")
    ap.add_argument("--weight", choices=["log", "sqrt", "none"], default="log",
                    help="loss weight per page: 1+log(1+views) (as XGBoost), sqrt(1+views), or equal")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--min-history", type=int, default=24, help="hours of data needed before a training 'now'")
    ap.add_argument("--out", default="models/transformer.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    M, hours, have, pages, test_start = load(args.data, args.top, args.test_days)
    n_h = len(hours)
    vs = n_h - args.val_hours                       # first validation target hour (index)
    print(f"training hours {hours[0]} .. {hours[-1]} ({n_h} h, test period starts {test_start}, never used); "
          f"{len(pages):,} pages; validation targets from {hours[vs] if args.val_hours else 'none'}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    Mp = np.concatenate([np.full((M.shape[0], CONTEXT - 1), np.nan, np.float32), M], axis=1)

    def usable(i):
        lo = max(0, i - 23)
        return have[i] and i >= args.min_history - 1 and have[lo:i + 1].mean() >= 0.75

    # ---- training samples: (page, now) pairs; targets after the validation start are masked
    train_nows = [i for i in range(n_h - 1) if usable(i) and i + 1 < vs]
    rows, nows = [], []
    for i in train_nows:
        W = Mp[:, i + CONTEXT - SELECT_WINDOW:i + CONTEXT]
        top = select_candidates(W, args.top_rows)
        rest = np.setdiff1d(np.nonzero(np.nan_to_num(W).sum(axis=1) > 0)[0], top)
        extra = rng.choice(rest, size=min(args.random_rows, len(rest)), replace=False) if len(rest) else rest
        idx = np.concatenate([top, extra])
        rows.append(idx); nows.append(np.full(len(idx), i))
    rows, nows = np.concatenate(rows), np.concatenate(nows)
    H = np.array(HORIZONS)
    tgt_idx = nows[:, None] + H[None, :]
    limit = vs if args.val_hours else n_h
    ok = tgt_idx < limit
    tgt_idx_c = np.minimum(tgt_idx, n_h - 1)
    ok &= have[tgt_idx_c]
    print(f"{len(train_nows)} training hours, {len(rows):,} (page, hour) samples  [{time.time() - t0:.0f}s]", flush=True)

    # ---- validation hours
    val_nows = [i for i in range(vs - 1, n_h - 1) if usable(i)] if args.val_hours else []
    val_cache = []
    for i in val_nows:
        W25 = Mp[:, i + CONTEXT - SELECT_WINDOW:i + CONTEXT]
        cand = select_candidates(W25, CANDIDATES)
        tok, g, lev = encode(windows(Mp, cand, np.full(len(cand), i)), hours[i])
        val_cache.append((i, cand, tok, g, lev))

    model = ARCHS[args.arch](d=args.d, layers=args.layers).to(DEVICE)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()):,} parameters on {DEVICE}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (min(args.samples_per_epoch, len(rows)) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)

    def validate():
        """Mean hit rate per (horizon, size) for the model, last_hour and oracle on validation hours."""
        res = {k: {} for k in ("model", "last_hour", "oracle")}
        for i, cand, tok, g, lev in val_cache:
            P = predict_log(model, tok, g, lev)
            for j, h in enumerate(HORIZONS):
                t = i + h
                if t >= n_h or not have[t]:
                    continue
                actual = M[:, t]
                for name, pr in (("model", P[:, j]), ("last_hour", lev), ("oracle", np.log1p(actual[cand]))):
                    hr = hit_rates(pr, cand, actual, args.sizes)
                    res[name].setdefault(h, []).append(hr)
        return {k: {h: np.mean(v, axis=0) for h, v in d.items()} for k, d in res.items()}

    def summarise(r):
        lines, gains = [], []
        for h in HORIZONS:
            if h not in r["model"]:
                continue
            m, b, o = r["model"][h], r["last_hour"][h], r["oracle"][h]
            gains += list(m - b)
            parts = [f"N={n:>6}: {mm:.4f} ({(mm - bb) * 100:+.2f} pts, {(mm - bb) / max(oo - bb, 1e-9):+.0%} of gap)"
                     for n, mm, bb, oo in zip(args.sizes, m, b, o)]
            lines.append(f"    h={h}: " + " | ".join(parts))
        return float(np.mean(gains)) if gains else 0.0, "\n".join(lines)

    best, best_state, history = -1e9, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(rows))[:args.samples_per_epoch]
        tot = cnt = 0.0
        for s in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[s:s + args.batch]
            W = windows(Mp, rows[b], nows[b])
            tok, g, lev = encode(W, hours[nows[b]])
            y = np.log1p(np.nan_to_num(M[rows[b][:, None], tgt_idx_c[b]])) - lev[:, None]
            v_now = np.nan_to_num(W[:, -1])
            w = {"log": 1 + np.log1p(v_now), "sqrt": np.sqrt(1 + v_now), "none": np.ones_like(v_now)}[args.weight]
            w = (w[:, None] * ok[b]).astype(np.float32)
            out = model(torch.from_numpy(tok).to(DEVICE), torch.from_numpy(g).to(DEVICE))
            y_t = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
            loss = (torch.from_numpy(w).to(DEVICE) * (out - y_t) ** 2).sum() / max(float(w.sum()), 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b); cnt += len(b)
        msg = f"epoch {ep}: train loss {tot / max(cnt, 1):.4f}  [{time.time() - t0:.0f}s]"
        if val_cache:
            r = validate()
            gain, table = summarise(r)
            msg += f"\n  validation: mean gain over last_hour {gain * 100:+.3f} pts\n{table}"
            history.append({"epoch": ep, "gain_pts": gain * 100})
            if gain > best:
                best, best_state = gain, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best epoch by validation hit rate: {max(history, key=lambda x: x['gain_pts'])['epoch']} "
              f"({best * 100:+.3f} pts over last_hour)")
    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save({"arch": args.arch, "kwargs": {"d": args.d, "layers": args.layers},
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(args), "history": history,
                "train_end": str(hours[-1])}, args.out)
    print(f"saved {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
