# Key numbers (test period: 72 hours the models never saw)

## Cache with room for 10,000 pages, deciding 1 hour ahead
- Perfect foresight (ceiling): **41.66%** of views served from cache (95% interval 41.32%–42.05%)
- XGBoost + decay feature: **40.59%** of views served from cache (95% interval 40.29%–40.92%)
- Last hour (standard rule): **40.47%** of views served from cache (95% interval 40.17%–40.81%)
- Physics: decay model: **40.28%** of views served from cache (95% interval 39.99%–40.61%)
- 24-hour average: **39.12%** of views served from cache (95% interval 38.89%–39.38%)
- Same hour yesterday: **36.66%** of views served from cache (95% interval 36.43%–36.93%)
- The standard rule leaves **1.19 points** on the table vs perfect foresight.

## Same-hours comparison with the standard rule (paired, 95% block-bootstrap intervals)
- 1 h ahead, Perfect foresight (ceiling): **+1.19 points** (95% interval +1.10 to +1.28, significant); better in 100% of hours
- 1 h ahead, XGBoost + decay feature: **+0.12 points** (95% interval +0.10 to +0.14, significant); better in 94% of hours
- 1 h ahead, Physics: decay model: **-0.19 points** (95% interval -0.22 to -0.15, significant); better in 0% of hours
- 6 h ahead, Perfect foresight (ceiling): **+4.20 points** (95% interval +3.60 to +4.61, significant); better in 100% of hours
- 6 h ahead, XGBoost + decay feature: **+1.28 points** (95% interval +0.93 to +1.49, significant); better in 100% of hours
- 6 h ahead, Physics: decay model: **-0.11 points** (95% interval -0.16 to -0.06, significant); better in 19% of hours

## Share of that gap each model recovers (1 hour ahead)
- 1,000 pages: Physics: decay model **-34%**, XGBoost + decay feature **+0%**
- 10,000 pages: Physics: decay model **-16%**, XGBoost + decay feature **+10%**
- 100,000 pages: Physics: decay model **+8%**, XGBoost + decay feature **+30%**

## Foresight (room for 10,000 pages): hit-rate points above the standard rule
- 1 h ahead: Physics: decay model -0.19, XGBoost + decay feature +0.12, Perfect foresight (ceiling) +1.19
- 2 h ahead: Physics: decay model -0.26, XGBoost + decay feature +0.24, Perfect foresight (ceiling) +1.95
- 3 h ahead: Physics: decay model -0.20, XGBoost + decay feature +0.50, Perfect foresight (ceiling) +2.66
- 6 h ahead: Physics: decay model -0.12, XGBoost + decay feature +1.27, Perfect foresight (ceiling) +4.19

## Spikes
- Median hours late catching a spike: Perfect foresight (ceiling) 0 h, XGBoost + decay feature 0 h, Last hour (standard rule) 0 h, Physics: decay model 0 h, 24-hour average 1 h, Same hour yesterday 14 h
- Share of spike pages' views (first 6 h of each spike) served from cache: Perfect foresight (ceiling) 99.6%, Last hour (standard rule) 93.5%, XGBoost + decay feature 92.8%, Physics: decay model 91.8%, 24-hour average 84.6%, Same hour yesterday 47.3%
- Example: **Tom Skerritt** peaked at 97,628 views/hour; the standard rule had it cached 0 h after the spike, XGBoost + decay feature after 0 h.

## Physics of attention
- After a spike, extra attention fades like a **power law** (alpha = 1.79, tau = 0.87 h), fitted on 1429 spikes from the training days.
- **Half-life of attention: about 25 minutes** after the peak (decay phase fitted on the first 10 h).
- **Daily echo:** 23 h after the peak, extra attention is back up to 2.0% of the peak, 1.5x its low point.
