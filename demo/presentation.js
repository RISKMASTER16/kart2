'use strict';

function renderPresentation(data) {
  const profile = resultProfile(data.cache_size);
  const sampleHours = profile?.paired_comparisons.find(item => item.horizon === 1)?.n_hours;
  element('summary-title').textContent = sampleHours ? `One hour ahead. Averaged across ${sampleHours} test hours.` : 'One hour ahead. Supplied test averages.';
  const content = element('summary-values');
  content.replaceChildren();
  for (const key of data.predictors) {
    const value = data.summary?.hit_rate?.[key];
    const card = node('div', undefined, 'summary-item');
    card.append(node('dt', modelLabel(data, key)), node('dd', isRate(value) ? percent.format(value) : 'Unavailable'), node('p', 'average hourly share of views the cache would serve'));
    content.append(card);
  }
  const oracle = data.summary?.oracle_hit_rate;
  const card = node('div', undefined, 'summary-item oracle-item');
  card.append(node('dt', 'Perfect hindsight'), node('dd', isRate(oracle) ? percent.format(oracle) : 'Unavailable'));
  element('oracle-values').replaceChildren(card);
  element('oracle-note').textContent = isRate(oracle)
    ? 'Perfect hindsight chooses articles using what actually happened later. It is the best possible score for this test, not a method we could use in advance. The technical name is the oracle.'
    : 'No perfect-hindsight reference is available for this file.';
  element('run-note').textContent = data._fake === true ? 'Synthetic development fixture; no measured performance claims.' : provenanceMessage;
  const capacity = isCount(data.cache_size) && data.cache_size > 0 ? number.format(data.cache_size) : 'an unspecified number of';
  const model = typeof data.our_model_name === 'string' && data.our_model_name.trim() ? ` Our forecast uses ${data.our_model_name}.` : '';
  element('summary-scope').textContent = `Both approaches have space for ${capacity} articles and choose one hour ahead. Percentages are test-period averages over the retained top 150,000 English Wikipedia articles per hour. They are separate from the selected replay hour.${model}`;
  const baseline = data.summary?.hit_rate?.last_hour;
  const forecast = data.summary?.hit_rate?.our_model;
  const explanation = element('result-explainer');
  if (isRate(baseline) && isRate(forecast)) {
    const delta = (forecast - baseline) * 100;
    explanation.replaceChildren(node('strong', `About ${Math.round(Math.abs(delta) * 10)} ${delta >= 0 ? 'more' : 'fewer'} views ready for every 1,000.`),
      node('p', `The forecast is equivalent to about ${Math.round(forecast * 1000)} views served from fast storage, compared with ${Math.round(baseline * 1000)} using the standard rule. These are rounded illustrations of the average percentages above.`));
  } else explanation.textContent = 'Some comparison results are missing. We show only the values that were supplied.';
  const findings = element('published-findings');
  findings.replaceChildren();
  if (profile && data._fake !== true) {
    const one = profile.paired_comparisons.find(item => item.horizon === 1);
    const six = profile.paired_comparisons.find(item => item.horizon === 6);
    element('result-takeaway').textContent = one && six && six.difference_pp > one.difference_pp && one.difference_pp > 0
      ? 'Our model made better cache decisions in this test. Its advantage grew when both methods had to choose articles six hours ahead.'
      : 'Here is what the two approaches achieved in the confirmed historical test.';
    for (const finding of profile.paired_comparisons) {
      const item = node('div', undefined, 'finding');
      item.append(node('h3', `Choosing ${finding.horizon} ${finding.horizon === 1 ? 'hour' : 'hours'} ahead`), node('p', `${finding.difference_pp >= 0 ? '+' : ''}${finding.difference_pp.toFixed(2)}`, 'finding-value'), node('p', `percentage points ${finding.difference_pp < 0 ? 'fewer' : 'more'} views served than with the standard rule`));
      const position = profile.summary.foresight_curve.horizons.indexOf(finding.horizon);
      const outcomes = node('dl', undefined, 'horizon-rates');
      for (const [key, label] of [['last_hour','Standard rule'],['our_model','Our model']]) {
        const rate = profile.summary.foresight_curve.hit_rate[key][position];
        const pair = node('div');
        pair.append(node('dt', label), node('dd', isRate(rate) ? percent.format(rate) : 'Unavailable'));
        outcomes.append(pair);
      }
      item.append(outcomes);
      findings.append(item);
    }
    element('findings-note').textContent = profile.paired_comparisons.map(item => `${item.horizon}h: ${item.n_hours} paired hours; difference ${item.difference_pp.toFixed(2)} percentage points, 95% block bootstrap interval ${item.ci_low_pp.toFixed(2)} to ${item.ci_high_pp.toFixed(2)}; better in ${item.better_hours_percent}% of evaluated hours.`).join(' ') + ` These are Computing’s published paired comparisons with space for ${number.format(data.cache_size)} articles. Rounded summary subtraction can differ by 0.01 points. No statistical test is run in the frontend.`;
  } else {
    element('result-takeaway').textContent = data._fake === true ? 'This is a synthetic preview. It shows how the experiment works, not a measured performance advantage.' : 'The results are available to explore, but this file has not been confirmed as the team’s measured run.';
    element('findings-note').textContent = 'Published comparisons appear only when the data matches the confirmed result bundle.';
  }
}

document.getElementById('pitch-start').addEventListener('click', () => {
  showAllPages = false;
  document.querySelectorAll('details').forEach(details => { details.open = false; });
  if (replay) selectHour(0);
  showView('start');
});
