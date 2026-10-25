# Three minute walkthrough

Use Start here, Explore a day, then What we found. The optional storage page
answers a second question. Keep it for questions if time is short.

| Time | Action | Explanation |
|---|---|---|
| 0:00 to 0:35 | Start here | A cache is limited fast storage. We compare keeping recently popular articles with forecasting what people will read next. This is a historical test. |
| 0:35 to 1:10 | Explore a day | Show two hours and pause. The percentages use the measured 100,000 space scores for each hour. The list shows what people read, not which articles were stored. |
| 1:10 to 2:00 | What we found | With 100,000 spaces, the standard rule serves 85.22% and our model 86.61% of sampled views. In rounded terms, that is about 14 more ready views per 1,000. Same capacity, different choices. |
| 2:00 to 2:30 | Planning ahead | Published paired gains: +1.38 percentage points at one hour and +3.08 at six hours. The advantage grows when choices must be made earlier. |
| 2:30 to 3:00 | Limits and close | Short evaluation period, hourly choices, equal article spaces and a retained top 150,000 page denominator. Spikes were already handled well by the standard rule. |

If asked whether more space helps, use Compare storage sizes. Keep Our model
selected and compare 40.59% at 10,000 with 86.61% at 100,000. Standard rule
can also be selected, giving 40.47% and 85.22%. Same approach, same planning time,
same evaluated hours; capacity changes. This is separate from the forecasting
question, where capacity stays fixed and the method changes.

Do not claim more storage is free or that this identifies the best cost tradeoff.
Article counts are not byte budgets. Per 1,000 counts are rounded equivalents
of average rates, not additional experiments.

Sources: results.csv and charts_100k/key_numbers.md. Published paired intervals
at 100k: +1.26 to +1.56 points at 1h; +2.76 to +3.35 at 6h. Sample counts at
1/2/3/6 hours are 72/71/70/67. The 1h rounded summary subtraction is +1.39,
while the paired report is +1.38. Show the supplied result for each metric.

Root demo.json still declares 10,000 spaces. Explore uses its capacity independent article demand and the separate 100k hourly.csv scores. Individual readiness tags are omitted because matching choices are not supplied. The website now shows surge trajectories as observations only. Catch-time markers and 10k comparisons have been removed. At 100k, spike traffic
served is 99.21% the standard rule and 99.06% forecast. Do not invent event causes;
checked editorial context remains separate in event-context.json.

Keep the three minute length until judging rules require otherwise. Technical
details stay collapsed unless requested. Rehearse on judging hardware and check
a local server reload without internet before freeze.


Explain the core pipeline aloud: both methods use only past views, select the
same number of articles, then are scored against actual views in the later hour.
Call the percentages cache hit rates, never forecast accuracy. On Explore they
refer to the selected hour; on Results they are averages over evaluated hours.

At six hours ahead, BOTH methods commit six hours before the target. The
standard rule reuses its latest observed ranking; the model predicts demand.
This establishes the value of planning ahead in the test, not early prediction
of a named spike. If asked about spikes, show the actual curve and state that
per-article six-hour advance selection has not been established by these files.
