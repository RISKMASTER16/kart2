"""Live cache planner: chooses the cache for the next hour from the latest published Wikipedia data.

Wikimedia publishes English Wikipedia page views once an hour, usually one to two hours after the
hour ends. This engine keeps the most recent hours in data/hours/ (fetching each new hour as soon as
it appears), then:

  1. ranks pages for the next hour with the trained XGBoost model and with the standard rule
     (last hour's most-read pages), and compares the two caches at 10,000 and 100,000 pages;
  2. scores both methods on recent hours that have already happened, using only data that was
     available before each hour, exactly as in backtest.py.

It runs inside api.py (see the /api/live endpoints) or from the command line:

    python live.py                                  # fetch the latest hours, print the plan
    python live.py --no-download                    # use only data already on disk
    python live.py --as-of "2026-10-22 20:00"       # replay: act as if that hour were the latest

Needs the trained models (python fit_decay.py && python train_xgb.py); without them only the
standard rule is shown.
"""
from __future__ import annotations

import argparse
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from download import BASE_URL, do_hour
from predictors import PREDICTORS

ROOT = Path(__file__).resolve().parent
HOUR = pd.Timedelta(hours=1)
KEEP_HOURS = 32          # 25 hours of features + recent hours to score
SCORE_HOURS = 6          # recent hours scored against what readers actually did
SIZES = (10_000, 100_000)
TOP_LIST = 25
DIFF_LIST = 15


def _utc_hour_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("h")


def _title(page: str) -> str:
    return page.replace("_", " ")


class LiveEngine:
    def __init__(self, data_dir: Path = ROOT / "data", download: bool = True, poll_seconds: int = 300,
                 top: int = 150_000, base_url: str = BASE_URL, as_of: pd.Timestamp | None = None,
                 workers: int = 3):
        self.data_dir = Path(data_dir)
        self.download = download and as_of is None
        self.poll_seconds = poll_seconds
        self.top = top
        self.base_url = base_url
        self.as_of = as_of
        self.workers = workers
        self._lock = threading.Lock()
        self._scores: dict[pd.Timestamp, dict] = {}
        self._state: dict = {"phase": "starting", "message": "Starting up…", "mode": self._mode(),
                             "plan": None, "scores": [], "hours_available": 0}
        self._thread: threading.Thread | None = None

    def _mode(self) -> str:
        return "replay" if self.as_of is not None else ("live" if self.download else "offline")

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="live-engine", daemon=True)
            self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def cycle(self) -> None:
        """Fetch any new hours, then recompute the plan and the recent scores."""
        try:
            if self.download:
                self._fetch_recent()
            history = self._load_history()
            if history.empty:
                self._set(phase="waiting", message="No hourly data yet. Waiting for downloads.")
                return
            self._compute(history)
        except Exception as e:  # keep serving the last good state
            self._set(phase="error", message=f"{type(e).__name__}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        while True:
            self.cycle()
            if self.as_of is not None:
                return
            self._set(next_check=(datetime.now(timezone.utc) + pd.Timedelta(seconds=self.poll_seconds)).isoformat())
            time.sleep(self.poll_seconds)

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)
            self._state["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _have(self) -> set[pd.Timestamp]:
        hours_dir = self.data_dir / "hours"
        return {pd.Timestamp(datetime.strptime(p.stem, "%Y%m%d-%H")) for p in hours_dir.glob("*.parquet")}

    def _fetch_recent(self) -> None:
        """Download the newest missing hours (newest first). An unpublished hour is skipped; a network
        failure stops this round and the engine keeps using what it has."""
        (self.data_dir / "hours").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        now = _utc_hour_now()
        have = self._have() | self._parquet_hours()
        wanted = [now - k * HOUR for k in range(1, KEEP_HOURS + 3)]
        missing = [h for h in wanted if h not in have]
        if not missing:
            return
        done = 0
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(do_hour, h, self.base_url, self.data_dir, self.top): h for h in missing}
            for fut in as_completed(futures):
                done += 1
                self._set(phase="downloading",
                          message=f"Fetching recent hours from Wikimedia ({done}/{len(missing)} checked)")
                try:
                    fut.result()
                except FileNotFoundError:
                    continue                                # not published yet
                except Exception as e:
                    self._set(mode="offline",
                              message=f"Wikimedia unreachable ({type(e).__name__}); using data on disk")
                    for f in futures:
                        f.cancel()
                    return
                self._set(mode="live")
                if done in (1, 4) or done % 8 == 0:         # show something useful while filling the window
                    history = self._load_history()
                    if not history.empty:
                        self._compute(history)

    def _parquet_hours(self) -> set[pd.Timestamp]:
        path = self.data_dir / "views.parquet"
        if not path.exists():
            return set()
        if not hasattr(self, "_pq_hours"):
            self._pq_hours = set(pd.DatetimeIndex(pd.read_parquet(path, columns=["hour"])["hour"].unique()))
        return self._pq_hours

    def _load_history(self) -> pd.DataFrame:
        """The most recent KEEP_HOURS hours (up to as_of when replaying), from data/hours/ and
        data/views.parquet, top pages per hour, sorted by hour as the predictors require."""
        frames = []
        hours_dir = self.data_dir / "hours"
        shard_hours = sorted(self._have())
        candidates = sorted(set(shard_hours) | self._parquet_hours())
        if self.as_of is not None:
            candidates = [h for h in candidates if h <= self.as_of]
        if not candidates:
            return pd.DataFrame(columns=["hour", "page", "views"])
        latest = candidates[-1]
        lo = latest - (KEEP_HOURS - 1) * HOUR
        from_shards = [h for h in shard_hours if lo <= h <= latest]
        for h in from_shards:
            frames.append(pd.read_parquet(hours_dir / f"{h:%Y%m%d-%H}.parquet", columns=["hour", "page", "views"]))
        rest = [h for h in candidates if lo <= h <= latest and h not in set(from_shards)]
        if rest:
            frames.append(pd.read_parquet(self.data_dir / "views.parquet", columns=["hour", "page", "views"],
                                          filters=[("hour", ">=", min(rest)), ("hour", "<=", max(rest))]))
        df = pd.concat(frames, ignore_index=True)
        df["hour"] = pd.to_datetime(df["hour"])
        df = df.drop_duplicates(["hour", "page"])
        df = (df.sort_values(["hour", "views"], ascending=[True, False], kind="stable")
                .groupby("hour", sort=False).head(self.top))
        return df.sort_values(["hour", "page"], kind="stable").reset_index(drop=True)

    def _predict(self, history: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.Series, pd.Series | None]:
        hist = history[history["hour"] <= now]
        standard = PREDICTORS["last_hour"](hist, now, 1)
        model = PREDICTORS["xgboost"](hist, now, 1) if "xgboost" in PREDICTORS else None
        return standard, model

    def _compute(self, history: pd.DataFrame) -> None:
        hours = pd.DatetimeIndex(sorted(history["hour"].unique()))
        latest = hours[-1]
        self._set(phase="computing", message=f"Choosing the cache for {latest + HOUR:%H:00} UTC…")
        by_hour = {h: g.set_index("page")["views"] for h, g in history.groupby("hour")}
        standard, model = self._predict(history, latest)
        ranked = model if model is not None else standard
        now_views = by_hour[latest]

        std_rank = pd.Series(np.arange(1, len(standard) + 1), index=standard.sort_values(ascending=False).index)
        top = ranked.sort_values(ascending=False).head(TOP_LIST)
        top_rows = [{"page": p, "title": _title(p), "views_now": int(now_views.get(p, 0)),
                     "predicted": round(float(v)), "standard_rank": int(std_rank.get(p, 0)) or None}
                    for p, v in top.items()]

        caches = {}
        for n in SIZES:
            s_top = set(standard.nlargest(n).index)
            m_top = set(ranked.nlargest(n).index)
            adds = ranked[ranked.index.isin(m_top - s_top)].sort_values(ascending=False).head(DIFF_LIST)
            drops = standard[standard.index.isin(s_top - m_top)].sort_values(ascending=False).head(DIFF_LIST)
            caches[str(n)] = {
                "overlap": round(1 - len(m_top - s_top) / max(len(m_top), 1), 4),
                "differ": len(m_top - s_top),
                "adds": [{"page": p, "title": _title(p), "views_now": int(now_views.get(p, 0)),
                          "predicted": round(float(v))} for p, v in adds.items()],
                "drops": [{"page": p, "title": _title(p), "views_now": int(now_views.get(p, 0)),
                           "predicted": round(float(ranked.get(p, 0)))} for p, v in drops.items()],
            }

        # Score recent hours that have happened: each method only sees data before that hour.
        for t in hours[-SCORE_HOURS:]:
            if t in self._scores or (t - HOUR) not in by_hour or model is None:
                continue
            s_pred, m_pred = self._predict(history, t - HOUR)
            actual = by_hour[t]
            total = float(actual.sum())
            row = {"hour": t.isoformat() + "Z", "total_views": int(total)}
            for n in SIZES:
                for name, pred in (("standard", s_pred), ("model", m_pred)):
                    cached = pred.nlargest(n).index
                    row[f"{name}_{n}"] = round(float(actual.reindex(cached).fillna(0).sum()) / total, 4)
            self._scores[t] = row

        latest_age = (_utc_hour_now() - latest) / HOUR if self.as_of is None else 0
        self._set(
            phase="ready",
            message="Ready",
            mode=self._mode() if self._state.get("mode") != "offline" else "offline",
            model=("xgboost" if model is not None else None),
            latest_hour=latest.isoformat() + "Z",
            target_hour=(latest + HOUR).isoformat() + "Z",
            latest_age_hours=round(float(latest_age), 1),
            hours_available=len(hours),
            history_hours=int(min(len(hours), 25)),
            total_views_latest=int(now_views.sum()),
            plan={"top": top_rows, "caches": caches},
            scores=[self._scores[t] for t in sorted(self._scores) if t in by_hour][-SCORE_HOURS:],
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true", help="use only data already on disk")
    ap.add_argument("--as-of", default=None, help='replay: treat this UTC hour as the latest, e.g. "2026-10-22 20:00"')
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    eng = LiveEngine(Path(args.data), download=not args.no_download, as_of=as_of)
    eng.cycle()
    s = eng.snapshot()
    if s.get("phase") != "ready":
        raise SystemExit(s.get("message"))
    print(f"Latest data: {s['latest_hour']} ({s['hours_available']} hours available, mode: {s['mode']})")
    print(f"Cache plan for {s['target_hour']}, top {TOP_LIST} by {'XGBoost' if s['model'] else 'the standard rule'}:")
    for i, r in enumerate(s["plan"]["top"], 1):
        print(f"  {i:2}. {r['title'][:50]:50} now {r['views_now']:>8,}  predicted {r['predicted']:>8,}")
    for n, c in s["plan"]["caches"].items():
        print(f"\n{int(n):,}-page cache: {c['differ']:,} pages differ from the standard rule "
              f"({c['overlap']:.1%} overlap). Our model adds: " + ", ".join(a["title"] for a in c["adds"][:5]))
    for r in s["scores"]:
        print(f"  scored {r['hour']}: " + "  ".join(f"{n:,}: standard {r[f'standard_{n}']:.2%} vs model {r[f'model_{n}']:.2%}"
                                                    for n in SIZES))


if __name__ == "__main__":
    main()
