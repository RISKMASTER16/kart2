'use strict';

let spikeRecords = [];

let eventContext = {};

async function loadEventContext(signal) {
  eventContext = {};
  try {
    const response = await fetch('event-context.json', { cache: 'no-store', signal });
    if (response.ok) eventContext = (await response.json()).events || {};
  } catch (error) { if (error.name === 'AbortError') throw error; }
}

function initializeSpikes(data) {

  const supplied = Array.isArray(data.spikes) ? data.spikes : [];
  spikeRecords = supplied.filter(record => record && typeof record.page === 'string' && record.page.trim()
    && isHour(record.spike_hour) && record.spike_hour.slice(0, 10) === data.day
    && Array.isArray(record.hours) && record.hours.length > 0
    && Array.isArray(record.views) && record.views.length === record.hours.length
    && record.hours.every(isHour) && new Set(record.hours).size === record.hours.length
    && record.hours.includes(record.spike_hour));
  const select = element('spike-select');
  select.replaceChildren();
  spikeRecords.forEach((record, index) => {
    const option = node('option', `${record.page.replaceAll('_', ' ')} · ${record.spike_hour.slice(11, 16)} UTC`);
    option.value = index;
    select.append(option);
  });
  select.disabled = spikeRecords.length === 0;
  element('spike-content').hidden = spikeRecords.length === 0;
  const mainProfile = resultProfile(100000);
  element('spike-plain').textContent = mainProfile && data._fake !== true
    ? `With space for 100,000 articles, both approaches served almost all spike views. The standard rule served ${percent.format(mainProfile.spike_hit_rate.last_hour)} and our model served ${percent.format(mainProfile.spike_hit_rate.our_model)}. These results do not establish that the model predicted a spike before it began.`
    : 'A sudden surge is called a spike. The recorded curves below show actual article demand, not a forecast.';
  const omitted = supplied.length - spikeRecords.length;
  element('spike-status').textContent = !spikeRecords.length
    ? 'No usable spike records supplied. Replay and foresight results remain available.'
    : `${spikeRecords.length} recorded article trajectories. View counts are actual observations and do not depend on cache size.${omitted ? ` ${omitted} malformed record(s) omitted.` : ''} Event classification comes from Computing; the prior 24 hour median cannot be verified from this short window.`;
  if (spikeRecords.length) renderSpike(0);
}

function renderSpike(index) {
  const record = spikeRecords[index];
  if (!record) return;
  const onset = Date.parse(record.spike_hour);
  const points = record.hours.map((hour, i) => ({ hour, time: Date.parse(hour), views: record.views[i] })).sort((a, b) => a.time - b.time);
  const atSpike = points.find(point => point.time === onset);
  const conforms = isCount(atSpike.views) && atSpike.views >= 1000;
  const warnings = [];
  if (!conforms) warnings.push('The supplied spike-hour count is unavailable or below the agreed 1,000-view minimum. The reported classification should be checked.');
  if (points.some(point => !isCount(point.views) || point.views === 0)) warnings.push('Zero or invalid counts are shown as gaps: zero may represent missing or outside recorded coverage data.');
  element('spike-warning').textContent = warnings.join(' ');
  element('spike-warning').hidden = !warnings.length;
  element('spike-name').textContent = record.page.replaceAll('_', ' ');
  element('spike-time').textContent = `Supplied onset: ${dateTime.format(new Date(onset))} UTC`;
  const context = eventContext[record.page];
  const contextNode = element('spike-context');
  contextNode.replaceChildren();
  let source;
  try { source = new URL(context?.source_url); } catch { /* Optional editorial context. */ }
  if (context?.status === 'checked' && typeof context.text === 'string' && context.text.trim() && source?.protocol === 'https:') {
    contextNode.append(document.createTextNode(`Event context (editorial): ${context.text} `));
    const link = node('a', 'Source');
    link.href = source.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    contextNode.append(link);
  } else contextNode.textContent = 'Event cause not established here. The plot shows supplied attention counts, without attributing them to a news event.';
  const jumpIndex = replay.hours.findIndex(hour => hour.hour === record.spike_hour);
  element('spike-jump').disabled = jumpIndex < 0;
  element('spike-jump').textContent = jumpIndex < 0 ? 'Event hour unavailable in replay' : 'Show event hour in replay';

  const body = element('spike-values');
  body.replaceChildren();
  points.forEach(point => {
    const row = node('tr');
    const label = node('th', `${dateTime.format(new Date(point.time))} UTC${point.time === onset ? ' · onset' : ''}`);
    label.scope = 'row';
    row.append(label, node('td', !isCount(point.views) ? 'Unavailable' : point.views === 0 ? '0 supplied · observation uncertain' : number.format(point.views)));
    body.append(row);
  });
  drawSpike(points, onset);
}

function drawSpike(points, onset) {
  const svg = element('spike-chart');
  svg.replaceChildren();
  const start = points[0].time;
  const end = points[points.length - 1].time;
  const span = Math.max(end - start, 3600000);
  const max = Math.max(1, ...points.filter(p => isCount(p.views) && p.views > 0).map(p => p.views));
  const ceiling = Math.ceil(max / 4) * 4;
  const x = time => 80 + (time - start) / span * 610;
  const y = views => 240 - views / ceiling * 185;
  const height = 330;
  svg.setAttribute('viewBox', `0 0 760 ${height}`);
  function shape(tag, attrs = {}, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  svg.append(shape('title', { id: 'spike-chart-title' }, 'Actual recorded article views'),
    shape('desc', {}, 'This chart shows actual hourly views, not model predictions. The dotted line marks the reported surge onset. Gaps represent uncertain observations. Exact counts are available below.'));
  for (const fraction of [0, .5, 1]) {
    svg.append(shape('line', { x1: 80, x2: 690, y1: y(ceiling * fraction), y2: y(ceiling * fraction), stroke: '#ccd2c9' }),
      shape('text', { x: 68, y: y(ceiling * fraction) + 4, 'text-anchor': 'end' }, number.format(ceiling * fraction)));
  }
  svg.append(shape('text', { x: 80, y: 22 }, 'Views per supplied hour'),
    shape('line', { x1: x(onset), x2: x(onset), y1: 35, y2: 250, stroke: '#92502c', 'stroke-dasharray': '3 4' }),
    shape('text', { x: x(onset), y: 38, 'text-anchor': 'middle' }, 'Spike onset'));
  let path = '';
  let previous = null;
  points.forEach(point => {
    if (!isCount(point.views) || point.views === 0) { previous = null; return; }
    const connected = previous !== null && point.time - previous === 3600000;
    path += `${connected ? 'L' : 'M'} ${x(point.time)} ${y(point.views)} `;
    previous = point.time;
    const circle = shape('circle', { cx: x(point.time), cy: y(point.views), r: 4, fill: '#172a29' });
    circle.append(shape('title', {}, `${dateTime.format(new Date(point.time))} UTC: ${number.format(point.views)} views`));
    svg.append(circle);
  });
  svg.append(shape('path', { d: path, fill: 'none', stroke: '#172a29', 'stroke-width': 2.5, 'data-spike-trajectory': '' }));
  // Limit ticks while retaining real elapsed-hour spacing.
  points.filter((point, i) => i === 0 || i === points.length - 1 || point.time === onset || i % Math.ceil(points.length / 6) === 0)
    .forEach(point => svg.append(shape('text', { x: x(point.time), y: 266, 'text-anchor': 'middle' }, point.hour.slice(11, 16))));
  svg.append(shape('text', { x: 385, y: 290, 'text-anchor': 'middle' }, 'UTC · exact dates in the values table'));

}

document.getElementById('spike-select').addEventListener('change', event => renderSpike(Number(event.target.value)));
document.getElementById('spike-jump').addEventListener('click', () => {
  const record = spikeRecords[Number(element('spike-select').value)];
  if (!record || !replay) return;
  const index = replay.hours.findIndex(hour => hour.hour === record.spike_hour);
  if (index < 0) return;
  selectHour(index);
  showView('explore');
  element('timeline').focus({ preventScroll: true });
});
