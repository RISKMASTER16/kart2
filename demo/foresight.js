'use strict';

// Display names never replace the keys used to look up supplied results.
function modelLabel(data, key) {
  if (key === 'last_hour') return 'Standard rule';
  if (key === 'our_model') return typeof data.our_model_name === 'string' && data.our_model_name.trim()
    ? `Our model (${data.our_model_name === 'xgboost' ? 'XGBoost' : data.our_model_name.trim()})` : 'Our model';
  return key.replaceAll('_', ' ');
}

function renderForesight(data) {
  renderHorizonLab(data);
  const horizons = [1, 2, 3, 6];
  const curve = data.summary?.foresight_curve;
  const supplied = curve?.horizons;
  const validHorizons = Array.isArray(supplied) && supplied.length > 0
    && supplied.every(h => horizons.includes(h)) && new Set(supplied).size === supplied.length;
  const series = data.predictors.map(key => {
    const values = curve?.hit_rate?.[key];
    // A length mismatch makes positional alignment uncertain: reject that series.
    const aligned = validHorizons && Array.isArray(values) && values.length === supplied.length;
    return { key, label: modelLabel(data, key), values: horizons.map(h => {
      const value = aligned ? values[supplied.indexOf(h)] : undefined;
      return isRate(value) ? value : null;
    }) };
  });
  const count = series.reduce((sum, model) => sum + model.values.filter(v => v !== null).length, 0);
  element('foresight-status').textContent = count === 0
    ? 'Foresight results unavailable. Waiting for valid summary values from Computing.'
    : count < series.length * horizons.length
      ? 'Partial results: unavailable values are left blank in the plot and marked in the table. Lines do not bridge missing horizons.'
      : 'Average hourly cache hit rates from the test period. These are measured outcomes, not predicted view counts.';
  element('foresight-coverage').textContent = approvedRun && data._fake !== true
    ? `Evaluated hours by horizon: ${horizons.map(h => `${h}h: ${approvedRun.hours_by_horizon[h]}`).join(' · ')}. Methods share the same hours within a horizon; coverage differs across horizons.`
    : 'Evaluation counts are not confirmed for this dataset.';

  const head = node('tr');
  ['How far ahead?', ...series.map(s => s.label)].forEach(label => {
    const cell = node('th', label);
    cell.scope = 'col';
    head.append(cell);
  });
  element('foresight-head').replaceChildren(head);
  const body = element('foresight-body');
  body.replaceChildren();
  horizons.forEach((h, index) => {
    const row = node('tr');
    const label = node('th', `${h} ${h === 1 ? 'hour' : 'hours'}`);
    label.scope = 'row';
    row.append(label);
    series.forEach(s => row.append(node('td', s.values[index] === null ? 'Unavailable' : percent.format(s.values[index]))));
    body.append(row);
  });

  const svg = element('foresight-chart');
  svg.replaceChildren();
  svg.style.display = count === 0 ? 'none' : '';
  const legend = element('foresight-legend');
  legend.replaceChildren();
  if (!count) return;
  const palette = ['#202122', '#3366cc', '#92502c', '#345fa1'];
  const dashes = ['', '8 5', '2 5', '10 4 2 4'];
  const x = h => 65 + (h - 1) / 5 * 640;
  // Zoom to padded five-point bounds; keep absolute rates and disclose the scale.
  const rates = series.flatMap(model => model.values.filter(value => value !== null));
  let lower = Math.max(0, Math.floor((Math.min(...rates) - .01) / .05) * .05);
  let upper = Math.min(1, Math.ceil((Math.max(...rates) + .01) / .05) * .05);
  if (upper - lower < .1) {
    lower = Math.max(0, Math.min(lower, upper - .1));
    upper = Math.min(1, Math.max(upper, lower + .1));
  }
  const y = rate => 265 - (rate - lower) / (upper - lower) * 220;
  element('foresight-status').textContent += ` Chart scale: ${percent.format(lower)} to ${percent.format(upper)}. The vertical axis is zoomed to make differences easier to see; it does not start at zero.`;
  function shape(tag, attrs = {}, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  svg.append(shape('title', { id: 'foresight-chart-title' }, 'Cache hit rate by forecast horizon'),
    shape('desc', {}, `Zoomed hit-rate axis from ${percent.format(lower)} to ${percent.format(upper)} at one, two, three and six hours. Exact values and model names appear in the table below. Missing values are not interpolated.`));
  for (const rate of Array.from({length: 5}, (_, index) => lower + index * (upper - lower) / 4)) {
    svg.append(shape('line', { x1: 65, x2: 705, y1: y(rate), y2: y(rate), stroke: '#ccd2c9' }),
      shape('text', { x: 52, y: y(rate) + 4, 'text-anchor': 'end' }, percent.format(rate)));
  }
  horizons.forEach(h => svg.append(shape('text', { x: x(h), y: 288, 'text-anchor': 'middle' }, `${h}h`)));
  svg.append(shape('text', { x: 65, y: 22 }, 'Views served from fast storage · zoomed scale'), shape('text', { x: 385, y: 320, 'text-anchor': 'middle' }, 'How many hours before readers arrive?'));
  series.forEach((model, index) => {
    const color = palette[index % palette.length];
    const dash = dashes[index % dashes.length];
    const entry = node('li');
    const swatch = node('span', index % 2 ? '■' : '●', 'series-swatch');
    swatch.style.color = color;
    swatch.setAttribute('aria-hidden', 'true');
    entry.append(swatch, document.createTextNode(`${model.label} · ${dash ? 'dashed' : 'solid'}`));
    legend.append(entry);
    let path = '';
    let connected = false;
    model.values.forEach((value, i) => {
      if (value === null) { connected = false; return; }
      path += `${connected ? 'L' : 'M'} ${x(horizons[i])} ${y(value)} `;
      connected = true;
    });
    svg.append(shape('path', { d: path, fill: 'none', stroke: color, 'stroke-width': 2.5, 'stroke-dasharray': dash, 'data-series': model.key }));
    model.values.forEach((value, i) => {
      if (value === null) return;
      const point = index % 2
        ? shape('rect', { x: x(horizons[i]) - 4, y: y(value) - 4, width: 8, height: 8, fill: color })
        : shape('circle', { cx: x(horizons[i]), cy: y(value), r: 4, fill: color });
      point.append(shape('title', {}, `${model.label}: ${percent.format(value)} at ${horizons[i]} hours`));
      svg.append(point);
    });
  });
  highlightHorizon();
}


let selectedHorizon = 1;
let horizonData = null;
function renderHorizonLab(data) {
  horizonData = data;
  const profile = resultProfile(100000);
  const ready = Boolean(profile && data._fake !== true);
  document.querySelectorAll('[name="planning-horizon"]').forEach(input => {
    input.disabled = !ready;
    input.checked = Number(input.value) === selectedHorizon;
  });
  element('planning-wait').textContent = `Wait ${selectedHorizon} ${selectedHorizon === 1 ? 'hour' : 'hours'}`;
  const output = element('horizon-outcome');
  output.replaceChildren();
  if (!ready) {
    output.append(node('p', 'The interactive comparison is unavailable until the published results match the confirmed run.'));
    return;
  }
  const curve = profile.summary.foresight_curve;
  const position = curve.horizons.indexOf(selectedHorizon);
  const scores = node('div', undefined, 'horizon-scores');
  for (const key of ['last_hour', 'our_model']) {
    const rate = curve.hit_rate[key][position];
    const card = node('div', undefined, 'horizon-score');
    card.append(node('h3', modelLabel(data, key)), node('p', percent.format(rate), 'interactive-rate'), node('p', 'average hourly views served', 'metric-note'));
    const meter = node('div', undefined, 'interactive-meter');
    meter.setAttribute('role', 'meter');
    meter.setAttribute('aria-label', `${modelLabel(data, key)}, ${selectedHorizon} hours ahead`);
    meter.setAttribute('aria-valuemin','0');
    meter.setAttribute('aria-valuemax','100');
    meter.setAttribute('aria-valuenow',String(rate * 100));
    const fill = node('span');
    fill.style.width = `${rate * 100}%`;
    meter.append(fill);
    card.append(meter);
    scores.append(card);
  }
  output.append(scores);
  const paired = profile.paired_comparisons.find(item => item.horizon === selectedHorizon);
  const difference = paired ? paired.difference_pp : (curve.hit_rate.our_model[position] - curve.hit_rate.last_hour[position]) * 100;
  output.append(node('p', `${difference >= 0 ? '+' : ''}${difference.toFixed(2)} percentage points with our model`, 'horizon-difference'));
  const hours = approvedRun.hours_by_horizon[selectedHorizon];
  output.append(node('p', `Both methods choose ${selectedHorizon} ${selectedHorizon === 1 ? 'hour' : 'hours'} before demand arrives. Compared across the same ${hours} evaluated hours. ${paired ? 'Difference from the published paired comparison.' : 'Difference calculated from the published rounded average rates.'}`, 'metric-note'));
  highlightHorizon();
}
function highlightHorizon() {
  const svg = element('foresight-chart');
  svg.querySelector('[data-horizon-highlight]')?.remove();
  if (!resultProfile(100000) || horizonData?._fake === true || !svg.querySelector('path')) return;
  const marker = document.createElementNS('http://www.w3.org/2000/svg','line');
  const x = 65 + (selectedHorizon - 1) / 5 * 640;
  for (const [name,value] of Object.entries({x1:x,x2:x,y1:40,y2:270,stroke:'#72777d','stroke-dasharray':'3 5','data-horizon-highlight':selectedHorizon})) marker.setAttribute(name,String(value));
  svg.insertBefore(marker,svg.querySelector('path'));
}
document.querySelectorAll('[name="planning-horizon"]').forEach(input => input.addEventListener('change', () => {
  if (!input.checked || !horizonData) return;
  selectedHorizon = Number(input.value);
  renderHorizonLab(horizonData);
}));
