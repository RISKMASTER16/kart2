# What the model is worth vs the standard rule (last_hour)

Test period: the 3 days the models never saw. 

## xgboost

| cache | ahead | hit rate (vs std) | fewer misses | gap closed | paired gain, 95% CI | hours won | 
|---|---|---|---|---|---|---|
| 1,000 | 1 h | 15.93% (15.93%) | **+0.0%** | 0% | +0.00 pts [-0.03, +0.01] | 41/72 (p=0.3) | 
| 1,000 | 2 h | 15.30% (15.26%) | **+0.0%** | 3% | +0.04 pts [-0.05, +0.12] | 52/71 (p=0.0001) | 
| 1,000 | 3 h | 14.96% (14.70%) | **+0.3%** | 13% | +0.26 pts [+0.14, +0.33] | 61/70 (p=1e-10) | 
| 1,000 | 6 h | 14.34% (13.62%) | **+0.8%** | 24% | +0.72 pts [+0.45, +0.92] | 64/67 (p=7e-16) | 
| 10,000 | 1 h | 40.59% (40.47%) | **+0.2%** | 10% | +0.12 pts [+0.10, +0.14] | 68/72 (p=5e-16) | 
| 10,000 | 2 h | 39.90% (39.66%) | **+0.4%** | 12% | +0.24 pts [+0.18, +0.28] | 70/71 (p=6e-20) | 
| 10,000 | 3 h | 39.42% (38.92%) | **+0.8%** | 19% | +0.50 pts [+0.39, +0.57] | 69/70 (p=1e-19) | 
| 10,000 | 6 h | 38.67% (37.40%) | **+2.0%** | 30% | +1.27 pts [+0.93, +1.49] | 67/67 (p=1e-20) | 
| 100,000 | 1 h | 86.61% (85.22%) | **+9.4%** | 30% | +1.39 pts [+1.25, +1.56] | 72/72 (p=4e-22) | 
| 100,000 | 2 h | 86.05% (84.42%) | **+10.5%** | 30% | +1.63 pts [+1.48, +1.84] | 71/71 (p=8e-22) | 
| 100,000 | 3 h | 85.65% (83.68%) | **+12.1%** | 32% | +1.97 pts [+1.79, +2.18] | 70/70 (p=2e-21) | 
| 100,000 | 6 h | 85.06% (81.98%) | **+17.1%** | 39% | +3.08 pts [+2.76, +3.36] | 67/67 (p=1e-20) | 

- Biggest cut in misses: **17.1% fewer requests to the main servers** (cache 100,000, 6 h ahead).
- Gain is statistically clear (paired 95% interval above zero) in 10 of 12 settings.
- Spikes (cache 10,000, 1 h ahead): 92.8% of spike pages' early views served from cache vs 93.5% for the standard rule; median catch lag 0 h vs 0 h.

## decay

| cache | ahead | hit rate (vs std) | fewer misses | gap closed | paired gain, 95% CI | hours won | 
|---|---|---|---|---|---|---|
| 1,000 | 1 h | 15.67% (15.93%) | **-0.3%** | -34% | -0.26 pts [-0.33, -0.21] | 2/72 (p=1e-18) | 
| 1,000 | 2 h | 15.02% (15.26%) | **-0.3%** | -17% | -0.24 pts [-0.31, -0.17] | 9/71 (p=7e-11) | 
| 1,000 | 3 h | 14.55% (14.70%) | **-0.2%** | -8% | -0.15 pts [-0.24, -0.09] | 18/70 (p=6e-05) | 
| 1,000 | 6 h | 13.58% (13.62%) | **-0.0%** | -1% | -0.04 pts [-0.18, +0.08] | 31/67 (p=0.6) | 
| 10,000 | 1 h | 40.28% (40.47%) | **-0.3%** | -16% | -0.19 pts [-0.22, -0.15] | 0/72 (p=4e-22) | 
| 10,000 | 2 h | 39.40% (39.66%) | **-0.4%** | -13% | -0.26 pts [-0.32, -0.20] | 4/71 (p=9e-16) | 
| 10,000 | 3 h | 38.72% (38.92%) | **-0.3%** | -8% | -0.20 pts [-0.24, -0.17] | 5/70 (p=2e-14) | 
| 10,000 | 6 h | 37.28% (37.40%) | **-0.2%** | -3% | -0.12 pts [-0.16, -0.06] | 13/67 (p=4e-07) | 
| 100,000 | 1 h | 85.57% (85.22%) | **+2.4%** | 8% | +0.35 pts [+0.27, +0.44] | 72/72 (p=4e-22) | 
| 100,000 | 2 h | 84.83% (84.42%) | **+2.6%** | 8% | +0.41 pts [+0.33, +0.52] | 70/71 (p=6e-20) | 
| 100,000 | 3 h | 84.14% (83.68%) | **+2.8%** | 8% | +0.46 pts [+0.37, +0.57] | 70/70 (p=2e-21) | 
| 100,000 | 6 h | 82.48% (81.98%) | **+2.8%** | 6% | +0.50 pts [+0.40, +0.62] | 67/67 (p=1e-20) | 

- Biggest cut in misses: **2.8% fewer requests to the main servers** (cache 100,000, 3 h ahead).
- Gain is statistically clear (paired 95% interval above zero) in 4 of 12 settings.
- Spikes (cache 10,000, 1 h ahead): 91.8% of spike pages' early views served from cache vs 93.5% for the standard rule; median catch lag 0 h vs 0 h.

How to read this:
- *Fewer misses*: misses are what load the main servers, so this is the saving that matters to an operator.
- *Paired gain*: model minus standard rule on the same hours; interval by block bootstrap (6-hour blocks).
- *Hours won*: hours where the model beat the standard rule; p from an exact sign test.
- *Std rule needs*: cache size at which the standard rule would match the model's hit rate.
- *Extra hits/day*: for this slice (English Wikipedia, top 150k pages per hour), not all Wikimedia traffic.