"""Combine views parquets (e.g. hours downloaded on several machines) into one data/views.parquet.

Accepts any mix of: .parquet files, folders (searched for *.parquet, e.g. data/hours/), and .zip
files containing parquets. Overlapping hours are fine: duplicate (hour, page) rows are merged.

What it does:
  - normalises column names (hour/page/views; common alternatives are recognised)
  - merges duplicates, keeping the larger view count if two sources disagree (and reports how often)
  - keeps the top --top pages per hour (default 150,000, as everywhere else)
  - reports the hour range, gaps (as ranges), and hours that look incomplete
  - backs up an existing output file before overwriting it
  - writes data/missing_hours.txt

Usage:
    python combine_parquets.py data/views.parquet other_views.parquet machine2.zip
    python combine_parquets.py data/hours/ ~/Downloads/views.zip --out data/views.parquet
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ALIASES = {
    "hour": ["hour", "timestamp", "datetime", "time", "date_hour"],
    "page": ["page", "title", "page_title", "article"],
    "views": ["views", "count", "count_views", "views_total", "pageviews"],
}


def find_files(inputs: list[str], tmp: Path) -> list[Path]:
    files = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            files += sorted(p.rglob("*.parquet"))
        elif p.suffix == ".zip":
            dest = tmp / p.stem
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            found = sorted(dest.rglob("*.parquet"))
            print(f"{p}: {len(found)} parquet file(s) inside")
            files += found
        elif p.suffix == ".parquet" and p.exists():
            files.append(p)
        else:
            raise SystemExit(f"not found or not a parquet/zip/folder: {item}")
    files = [f for f in files if not f.name.endswith(".tmp")]
    if not files:
        raise SystemExit("no parquet files found")
    return files


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {}
    for canon, names in ALIASES.items():
        match = next((c for c in df.columns if c.lower() in names), None)
        if match is None:
            raise SystemExit(f"{path}: no '{canon}' column (columns: {list(df.columns)})")
        rename[match] = canon
    df = df.rename(columns=rename)[["hour", "page", "views"]]
    df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_localize(None).dt.floor("h")
    df["page"] = df["page"].astype(str)
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    return df.dropna(subset=["views"])


def ranges(hours: list[pd.Timestamp]) -> list[str]:
    """Collapse a sorted list of hours into 'start .. end' ranges."""
    out, start, prev = [], None, None
    for h in hours:
        if start is None:
            start = prev = h
        elif h - prev == pd.Timedelta(hours=1):
            prev = h
        else:
            out.append(f"{start} .. {prev}" if start != prev else str(start))
            start = prev = h
    if start is not None:
        out.append(f"{start} .. {prev}" if start != prev else str(start))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="parquet files, folders, or zips")
    ap.add_argument("--out", default="data/views.parquet")
    ap.add_argument("--top", type=int, default=150_000, help="pages kept per hour (0 = keep all)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        files = find_files(args.inputs, Path(tmpdir))
        parts = []
        for f in files:
            df = load(f)
            parts.append(df)
            if len(files) <= 20:
                print(f"  {f.name}: {len(df):,} rows, {df['hour'].nunique()} hours, "
                      f"{df['hour'].min()} .. {df['hour'].max()}")
        print(f"read {len(files)} file(s)")
        df = pd.concat(parts, ignore_index=True)

    n_before = len(df)
    agg = df.groupby(["hour", "page"], sort=False)["views"].agg(["max", "min", "size"])
    conflicts = int(((agg["size"] > 1) & (agg["max"] != agg["min"])).sum())
    dupes = int((agg["size"] > 1).sum())
    df = agg["max"].rename("views").reset_index()
    print(f"merged {n_before:,} rows -> {len(df):,} ({dupes:,} page-hours appeared in more than one file"
          + (f", {conflicts:,} with different counts: kept the larger)" if conflicts else ", all identical)"))
    if conflicts > 0.05 * max(dupes, 1):
        print("  WARNING: the sources disagree a lot on overlapping hours. They may have been made differently "
              "(e.g. one without mobile views). Check before trusting the overlap.")

    if args.top:
        df = df.sort_values(["hour", "views"], ascending=[True, False]).groupby("hour", sort=False).head(args.top)
    df["views"] = df["views"].round().astype("int64")
    df = df.sort_values(["hour", "page"]).reset_index(drop=True)

    hours = pd.DatetimeIndex(df["hour"].unique()).sort_values()
    full = pd.date_range(hours.min(), hours.max(), freq="h")
    missing = [h for h in full if h not in set(hours)]
    per_hour = df.groupby("hour").size()
    thin = per_hour[per_hour < 0.5 * per_hour.median()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup = out.with_name(out.stem + "_backup.parquet")
        shutil.copy2(out, backup)
        print(f"backed up the old {out} to {backup}")
    df.to_parquet(out, index=False)
    (out.parent / "missing_hours.txt").write_text("".join(f"{h:%Y-%m-%d %H:%M:%S}\n" for h in missing))

    print(f"\nwrote {out}: {len(df):,} rows, {len(hours)} hours, {hours.min()} .. {hours.max()} "
          f"({(hours.max() - hours.min()) / pd.Timedelta(days=1):.1f} days)")
    print(f"rows per hour: median {per_hour.median():,.0f}, min {per_hour.min():,}, max {per_hour.max():,}")
    if missing:
        print(f"{len(missing)} missing hour(s): " + "; ".join(ranges(missing)[:10]) +
              (" ..." if len(ranges(missing)) > 10 else ""))
    else:
        print("no gaps: one continuous block")
    if len(thin):
        print(f"{len(thin)} hour(s) look incomplete (under half the usual rows): "
              + ", ".join(str(h) for h in thin.index[:10]))


if __name__ == "__main__":
    main()
