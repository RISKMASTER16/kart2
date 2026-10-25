"""
Data-quality checks on data/views.parquet.

    python checks.py

Prints a short report and writes:
  data/hour_totals.csv    total views per hour (spot partial or failed downloads)
  data/bot_suspects.csv   pages whose traffic looks automated

Bot heuristic: real readers follow the day/night cycle of English Wikipedia as a
whole. A page with lots of views that is almost flat across the day
(low coefficient of variation) AND doesn't follow the global daily curve is
probably a crawler or script. These pages are flagged, not removed, because excluding
them would change every downstream result.
"""

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant pages -> undefined correlation

df = pd.read_parquet("data/views.parquet")
hours = pd.date_range(df.hour.min(), df.hour.max(), freq="h")
print(f"{df.hour.nunique()} hours, {df.page.nunique():,} pages, {len(df):,} rows, "
      f"{df.hour.min()} .. {df.hour.max()} UTC")

# 1. Missing hours -----------------------------------------------------------
missing = hours.difference(pd.DatetimeIndex(df.hour.unique()))
print(f"\nMissing hours: {len(missing)}" + ("".join(f"\n  {h}" for h in missing) if len(missing) else ""))

# 2. Suspicious hour totals (partial files, outages) ------------------------
tot = df.groupby("hour").agg(views=("views", "sum"), pages=("page", "size"))
tot["how"] = tot.index.hour
typical = tot.groupby("how")["views"].transform("median")
tot["ratio_to_typical"] = (tot["views"] / typical).round(3)
tot.to_csv("data/hour_totals.csv")
odd = tot[(tot.ratio_to_typical < 0.7) | (tot.ratio_to_typical > 1.5)]
print(f"\nHours with total views <70% or >150% of typical for that hour of day: {len(odd)}")
if len(odd):
    print(odd[["views", "pages", "ratio_to_typical"]].to_string())

# 3. Bot-like pages ------------------------------------------------------------
if df.hour.nunique() >= 24:
    present = tot.index  # judge flatness only on hours we actually have
    global_curve = tot["views"]
    top = df.groupby("page")["views"].sum().nlargest(5000).index
    wide = (df[df.page.isin(top)]
            .pivot_table(index="hour", columns="page", values="views", fill_value=0)
            .reindex(present, fill_value=0))
    stats = pd.DataFrame({
        "total_views": wide.sum(),
        "mean_per_hour": wide.mean().round(1),
        "cv": (wide.std() / wide.mean()).round(3),           # flatness: low = robotic
        "corr_with_global": wide.corrwith(global_curve).fillna(0).round(3),  # perfectly flat = 0
        "active_hours_frac": (wide > 0).mean().round(3),
    })
    sus = stats[(stats.mean_per_hour >= 100) & (stats.active_hours_frac > 0.95)
                & (stats.cv < 0.15) & (stats.corr_with_global < 0.3)]
    sus = sus.sort_values("total_views", ascending=False)
    sus.to_csv("data/bot_suspects.csv")
    print(f"\nBot suspects among top 5,000 pages: {len(sus)} (data/bot_suspects.csv)")
    if len(sus):
        print(sus.head(15).to_string())

# 4. Biggest single-hour jumps (candidates for the spike story) --------------
d = df.sort_values(["page", "hour"])
d["prev"] = d.groupby("page")["views"].shift().fillna(0)
d["jump"] = d["views"] / (d["prev"] + 50)
print("\nBiggest hour-on-hour jumps (candidate spike examples):")
print(d[(d.views >= 1000) & (d.jump >= 5)].nlargest(15, "jump")[["hour", "page", "prev", "views"]].to_string(index=False))
