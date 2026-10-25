# Key numbers (test period: 72 hours the models never saw)

## Cache with room for 100,000 pages, deciding 1 hour ahead
- Perfect foresight (ceiling): **89.82%** of views served from cache (95% interval 89.71%–89.95%)
- XGBoost + decay feature: **86.61%** of views served from cache (95% interval 86.41%–86.82%)
- Physics: decay model: **85.57%** of views served from cache (95% interval 85.34%–85.81%)
- 24-hour average: **85.48%** of views served from cache (95% interval 85.27%–85.69%)
- Last hour (standard rule): **85.22%** of views served from cache (95% interval 84.96%–85.50%)
- Same hour yesterday: **82.16%** of views served from cache (95% interval 81.89%–82.45%)
- The standard rule leaves **4.60 points** on the table vs perfect foresight.

## Same-hours comparison with the standard rule (paired, 95% block-bootstrap intervals)
- 1 h ahead, Perfect foresight (ceiling): **+4.60 points** (95% interval +4.27 to +5.04, significant); better in 100% of hours
- 1 h ahead, XGBoost + decay feature: **+1.38 points** (95% interval +1.26 to +1.56, significant); better in 100% of hours
- 1 h ahead, Physics: decay model: **+0.34 points** (95% interval +0.27 to +0.44, significant); better in 100% of hours
- 6 h ahead, Perfect foresight (ceiling): **+7.83 points** (95% interval +7.28 to +8.26, significant); better in 100% of hours
- 6 h ahead, XGBoost + decay feature: **+3.08 points** (95% interval +2.76 to +3.35, significant); better in 100% of hours
- 6 h ahead, Physics: decay model: **+0.50 points** (95% interval +0.40 to +0.62, significant); better in 100% of hours

## Share of that gap each model recovers (1 hour ahead)
- 1,000 pages: Physics: decay model **-34%**, XGBoost + decay feature **+0%**
- 10,000 pages: Physics: decay model **-16%**, XGBoost + decay feature **+10%**
- 100,000 pages: Physics: decay model **+8%**, XGBoost + decay feature **+30%**

## Foresight (room for 100,000 pages): hit-rate points above the standard rule
- 1 h ahead: Physics: decay model +0.35, XGBoost + decay feature +1.39, Perfect foresight (ceiling) +4.60
- 2 h ahead: Physics: decay model +0.41, XGBoost + decay feature +1.63, Perfect foresight (ceiling) +5.39
- 3 h ahead: Physics: decay model +0.46, XGBoost + decay feature +1.97, Perfect foresight (ceiling) +6.12
- 6 h ahead: Physics: decay model +0.50, XGBoost + decay feature +3.08, Perfect foresight (ceiling) +7.82

## Spikes
- Median hours late catching a spike: Perfect foresight (ceiling) 0 h, XGBoost + decay feature 0 h, Physics: decay model 0 h, 24-hour average 0 h, Last hour (standard rule) 0 h, Same hour yesterday 0 h
- Share of spike pages' views (first 6 h of each spike) served from cache: Perfect foresight (ceiling) 100.0%, Last hour (standard rule) 99.2%, Physics: decay model 99.1%, XGBoost + decay feature 99.1%, 24-hour average 98.4%, Same hour yesterday 86.1%

## Physics of attention
- After a spike, extra attention fades like a **power law** (alpha = 1.79, tau = 0.87 h), fitted on 1429 spikes from the training days.
- **Half-life of attention: about 25 minutes** after the peak (decay phase fitted on the first 10 h).
- **Daily echo:** 23 h after the peak, extra attention is back up to 2.0% of the peak, 1.5x its low point.
