# XGBoost vs normal caching methods

**Question.** A cache can hold only the top N Wikipedia articles each hour. Can predicting next hour's readers beat the standard caching rule, which keeps whatever was most read last hour?

**Setup.** English Wikipedia (desktop + mobile), hourly page views. The last 72 hours are the test period; XGBoost was trained only on the hours before it and never saw them. Every method is scored on exactly the same hours, cache sizes (1,000 / 10,000 / 100,000 pages) and warning times (1, 2, 3, 6 hours ahead).

## Headline

- With a 100,000-page cache deciding 1 hour ahead, XGBoost sends **9.4% fewer requests to the main servers** than the standard rule, and wins **72 of 72 test hours**.
- Deciding 6 hours ahead (a cache that needs time to warm up), the cut grows to **17.1% fewer misses**, winning **67 of 67 hours**.
- XGBoost captures **30%–39%** of the improvement that perfect knowledge of the future would give at this cache size.
- That is about **1.6 million** (1 h ahead) to **3.5 million** (6 h ahead) extra page views served from cache per day, for the top 150,000 English Wikipedia pages alone.
- The gain over the standard rule is statistically clear in **10 of 12** settings, and XGBoost is significantly better than *every* normal method in **46 of 48** method-by-setting comparisons (significantly worse in 0).

![Fewer misses](1_fewer_misses.png)

## Why misses, not hit rate

A cache hit is cheap; a miss means the main servers must build the page. So the operator's cost scales with the miss rate. Going from 85.2% to 86.6% hit rate looks like +1.4 points, but it cuts misses from 14.8% to 13.4% of all views: **9.4% less load** on the servers behind the cache.

## Full results vs the standard rule

Paired gain = XGBoost minus the standard rule, hour by hour on the same hours, in hit-rate points, with a 95% interval. *Gap closed* = share of the standard rule's distance to perfect foresight that XGBoost recovers.

| Cache | Ahead | XGBoost hit rate | Standard rule | Fewer misses | Paired gain [95% CI] | Hours won | Sign-test p | Gap closed |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 h | 15.93% | 15.93% | **0.0%** | +0.00 [-0.03, +0.01] | 41/72 | 0.29 | 0% |
| 1,000 | 2 h | 15.30% | 15.26% | **0.0%** | +0.04 [-0.05, +0.12] | 52/71 | 1×10⁻⁴ | 3% |
| 1,000 | 3 h | 14.96% | 14.70% | **0.3%** | +0.26 [+0.14, +0.33] | 61/70 | 1×10⁻¹⁰ | 13% |
| 1,000 | 6 h | 14.34% | 13.62% | **0.8%** | +0.72 [+0.45, +0.92] | 64/67 | 7×10⁻¹⁶ | 24% |
| 10,000 | 1 h | 40.59% | 40.47% | **0.2%** | +0.12 [+0.10, +0.14] | 68/72 | 5×10⁻¹⁶ | 10% |
| 10,000 | 2 h | 39.90% | 39.66% | **0.4%** | +0.24 [+0.18, +0.28] | 70/71 | 6×10⁻²⁰ | 12% |
| 10,000 | 3 h | 39.42% | 38.92% | **0.8%** | +0.50 [+0.39, +0.57] | 69/70 | 1×10⁻¹⁹ | 19% |
| 10,000 | 6 h | 38.67% | 37.40% | **2.0%** | +1.27 [+0.93, +1.49] | 67/67 | 1×10⁻²⁰ | 30% |
| 100,000 | 1 h | 86.61% | 85.22% | **9.4%** | +1.39 [+1.25, +1.56] | 72/72 | 4×10⁻²² | 30% |
| 100,000 | 2 h | 86.05% | 84.42% | **10.5%** | +1.63 [+1.48, +1.84] | 71/71 | 8×10⁻²² | 30% |
| 100,000 | 3 h | 85.65% | 83.68% | **12.1%** | +1.97 [+1.79, +2.18] | 70/70 | 2×10⁻²¹ | 32% |
| 100,000 | 6 h | 85.06% | 81.98% | **17.1%** | +3.08 [+2.76, +3.36] | 67/67 | 1×10⁻²⁰ | 39% |

![Every hour](2_every_hour.png)

## The strictest test: XGBoost vs the best normal method in each setting

Which simple rule works best depends on the setting (for long warning times the 24-hour average beats "last hour"). Here XGBoost faces whichever normal method scored highest in each setting.

| Cache | Ahead | Best normal method | Its hit rate | XGBoost | Paired gain [95% CI] | Fewer misses | Hours won |
|---|---|---|---|---|---|---|---|
| 1,000 | 1 h | Last hour (standard rule) | 15.93% | 15.93% | +0.00 [-0.03, +0.01] | **0.0%** | 41/72 |
| 1,000 | 2 h | Last hour (standard rule) | 15.26% | 15.30% | +0.04 [-0.05, +0.12] | **0.0%** | 52/71 |
| 1,000 | 3 h | Last hour (standard rule) | 14.70% | 14.96% | +0.26 [+0.14, +0.33] | **0.3%** | 61/70 |
| 1,000 | 6 h | 24-hour average | 13.68% | 14.34% | +0.66 [+0.56, +0.74] | **0.8%** | 67/67 |
| 10,000 | 1 h | Last hour (standard rule) | 40.47% | 40.59% | +0.12 [+0.10, +0.14] | **0.2%** | 68/72 |
| 10,000 | 2 h | Last hour (standard rule) | 39.66% | 39.90% | +0.24 [+0.18, +0.28] | **0.4%** | 70/71 |
| 10,000 | 3 h | Last hour (standard rule) | 38.92% | 39.42% | +0.50 [+0.39, +0.57] | **0.8%** | 69/70 |
| 10,000 | 6 h | 24-hour average | 37.93% | 38.67% | +0.74 [+0.61, +0.86] | **1.2%** | 67/67 |
| 100,000 | 1 h | Decay model (physics) | 85.57% | 86.61% | +1.04 [+0.97, +1.13] | **7.2%** | 72/72 |
| 100,000 | 2 h | 24-hour average | 85.21% | 86.05% | +0.84 [+0.70, +0.96] | **5.7%** | 71/71 |
| 100,000 | 3 h | 24-hour average | 85.01% | 85.65% | +0.64 [+0.52, +0.76] | **4.3%** | 70/70 |
| 100,000 | 6 h | 24-hour average | 84.59% | 85.06% | +0.47 [+0.35, +0.54] | **3.0%** | 66/67 |

![All methods](4_misses_all_methods.png)

## What foresight is worth

If a cache needs k hours' notice to load an article, every method gets worse as k grows, but the standard rule falls fastest. Prediction matters most exactly when caches need the most notice.

| Ahead | Perfect foresight | XGBoost (ours) | 24-hour average | Last hour (standard rule) |
|---|---|---|---|---|
| 1 h | 89.82% | 86.61% | 85.48% | 85.22% |
| 2 h | 89.81% | 86.05% | 85.21% | 84.42% |
| 3 h | 89.80% | 85.65% | 85.01% | 83.68% |
| 6 h | 89.80% | 85.06% | 84.59% | 81.98% |

_Cache of 100,000 pages._

![Foresight](3_foresight.png)

## How we tested (so the numbers hold up)

- **No peeking:** XGBoost was trained only on hours before the 72-hour test period; at each test hour it sees only data up to that hour.
- **Paired comparison:** both methods are scored on the same hours, so hour-to-hour swings in traffic cancel out. The 95% interval resamples blocks of 6 consecutive hours (block bootstrap), because neighbouring hours are correlated; resampling single hours would overstate certainty.
- **Sign test:** counts the hours XGBoost won; p is the chance of a split at least that lopsided if the two methods were equally good.
- **Many comparisons:** XGBoost vs 4 normal methods x 12 settings = 48 comparisons. With a Bonferroni correction the bar is p < 0.0010; the largest sign-test p among the 46 significant wins is 1×10⁻¹⁰, so every one survives the correction.

## Honest limits

- **Small caches, short warning:** with 1,000 pages deciding 1 hour ahead, XGBoost ties the standard rule (+0.00 points, interval -0.03 to +0.01), and 2 hours ahead the gain is too small to be sure of. The most-read pages are easy for any method; the value is in the long tail and in deciding early.
- **Sudden spikes:** in the first 6 hours of a spike, XGBoost served 92.8% of the spiking pages' views from a 10,000-page cache, vs 93.5% for the standard rule: no better. A page that jumps from nothing within a single hour can't be foreseen from hourly data by any method.
- **Hourly decisions:** real caches decide per request; we decide once per hour.
- **Equal page sizes:** article sizes aren't in the data, so every page counts the same.
- **Scope:** 162 hours of English Wikipedia, top 150,000 pages per hour; savings are for this slice.

## Future work

- A transformer (sequence model) over each page's last 48 hours, trained on more weeks of data.
- Testing on Wikimedia's released per-request cache traces.