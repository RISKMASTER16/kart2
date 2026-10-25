'use strict';

// Consume teammate-owned output. Never infer performance from the top-50 sample.
const SOURCE = '../demo.json';
const FRAME_MS = 2000;
let replay = null;
let selectedIndex = 0;
let playbackTimer = null;
let showAllPages = false;
const element = id => document.getElementById(id);
const number = new Intl.NumberFormat('en-GB');
const percent = new Intl.NumberFormat('en-GB', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const dateTime = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'UTC', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
});
const isCount = value => Number.isSafeInteger(value) && value >= 0;
const isRate = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
const isHour = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:00:00Z$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value.replace('Z', '.000Z');

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

function validate(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The replay file must contain a JSON object.');
  if (typeof data.day !== 'string' || !isHour(`${data.day}T00:00:00Z`)) throw new Error('The replay day is missing or invalid.');
  if (!Array.isArray(data.predictors) || !data.predictors.length || data.predictors.some(name => typeof name !== 'string' || !name.trim())
    || new Set(data.predictors).size !== data.predictors.length) throw new Error('The dataset needs unique, non-empty model names.');
  if (!Array.isArray(data.hours) || !data.hours.length) throw new Error('No replay hours have been supplied.');
  const hours = data.hours.filter(hour => hour && isHour(hour.hour) && hour.hour.slice(0, 10) === data.day).sort((a, b) => a.hour.localeCompare(b.hour));
  if (!hours.length) throw new Error('No valid UTC hours are available for the replay day.');
  if (new Set(hours.map(hour => hour.hour)).size !== hours.length) throw new Error('Duplicate replay hours need to be corrected in the source file.');
  return { hours, omitted: data.hours.length - hours.length };
}

function render() {
  const { data, hours, omitted } = replay;
  const hour = hours[selectedIndex];
  const warnings = [];
  if (omitted) warnings.push(`${omitted} invalid or out-of-day hour(s) omitted.`);
  element('provenance').textContent = provenanceLabel(data);
  element('hour').dateTime = hour.hour;
  element('hour').textContent = `${dateTime.format(new Date(hour.hour))} UTC`;
  element('interval').textContent = `${hour.hour.slice(11, 16)} to ${hour.hour.slice(11, 13)}:59 UTC. Hour ${selectedIndex + 1} of ${hours.length} available.`;
  element('history-cutoff').textContent = `History through the hour starting ${dateTime.format(new Date(Date.parse(hour.hour) - 3600000))} UTC.`;
  element('observed-window').textContent = `Recorded views in the hour starting ${dateTime.format(new Date(hour.hour))} UTC.`;
  updateControls();
  element('capacity').textContent = `Space for ${number.format(replay.capacity)} articles`;
  element('table-hint').textContent = replay.hasCacheChoices
    ? '✓ Ready means already in fast storage. × Not ready means the main server would be needed.'
    : 'This list shows what people read. Individual article readiness is not included in the published 100,000 space results.';
  document.querySelector('.table-scroll').classList.toggle('demand-only', !replay.hasCacheChoices);
  element('total').textContent = isCount(hour.total_views) ? number.format(hour.total_views) : 'Unavailable';
  element('hour-count').textContent = `${hours.length} / 24`;
  if (!isCount(hour.total_views)) warnings.push('Some context values are unavailable.');
  const models = element('performance');
  models.replaceChildren();
  for (const name of data.predictors) {
    const value = hour.hit_rate?.[name];
    const valid = isRate(value);
    const card = node('section', undefined, 'model');
    const label = node('h3', modelLabel(data, name), 'model-label');
    label.append(node('span', name === 'last_hour' ? 'Rank using the latest observed hour' : 'Forecast demand from earlier page views'));
    const rate = node('p', valid ? percent.format(value) : 'Unavailable', `rate${valid ? '' : ' unavailable'}`);
    card.append(label, rate);
    if (valid) {
      const meter = node('div', undefined, 'meter');
      meter.setAttribute('aria-hidden', 'true');
      const fill = node('div', undefined, 'meter-fill');
      fill.style.width = `${value * 100}%`;
      meter.append(fill);
      card.append(meter);
    } else warnings.push(`Result unavailable for ${modelLabel(data, name)}.`);
    models.append(card);
  }
  const heading = node('tr');
  const standardRate = hour.hit_rate?.last_hour;
  const modelRate = hour.hit_rate?.our_model;
  element('hourly-gap').textContent = isRate(standardRate) && isRate(modelRate)
    ? `For this hour, our model’s cache would serve ${Math.abs((modelRate - standardRate) * 100).toFixed(2)} percentage points ${modelRate >= standardRate ? 'more' : 'fewer'} recorded views than the standard rule.`
    : 'The result difference is unavailable for this hour.';
  for (const [title, className] of [['#', ''], ['Article', ''], ['Views', 'views'], ...(replay.hasCacheChoices ? data.predictors.map(name => [modelLabel(data, name), 'cache']) : [])]) {
    const cell = node('th', title, className);
    cell.scope = 'col';
    heading.append(cell);
  }
  element('table-head').replaceChildren(heading);
  const suppliedPages = Array.isArray(hour.top_pages) ? hour.top_pages : [];
  const pages = suppliedPages.filter(page => page && typeof page.page === 'string' && page.page.trim() && isCount(page.views));
  const invalidPages = suppliedPages.length - pages.length;
  if (invalidPages) warnings.push(`${invalidPages} invalid article row(s) omitted.`);
  pages.sort((a, b) => b.views - a.views);
  const visiblePages = showAllPages ? pages : pages.slice(0, 6);
  element('page-count').textContent = `Showing ${visiblePages.length} of ${pages.length} recorded articles, most read first.`;
  element('show-pages').hidden = pages.length <= 6;
  element('show-pages').textContent = showAllPages ? 'Show fewer' : `Show all ${pages.length}`;
  element('show-pages').setAttribute('aria-expanded', String(showAllPages));
  const body = element('table-body');
  body.replaceChildren();
  let unknownCache = false;
  visiblePages.forEach((page, index) => {
    const row = node('tr');
    const title = node('th');
    title.append(node('span', page.page.replaceAll('_', ' '), 'article-title'), node('span', `${number.format(page.views)} views`, 'article-views-mobile'));
    title.scope = 'row';
    row.append(node('td', String(index + 1).padStart(2, '0'), 'rank'), title, node('td', number.format(page.views), 'views'));
    for (const name of replay.hasCacheChoices ? data.predictors : []) {
      const cached = page.cached?.[name];
      const known = typeof cached === 'boolean';
      unknownCache ||= !known;
      const cell = node('td', undefined, 'cache');
      cell.append(node('span', known ? (cached ? '✓ Ready' : '× Not ready') : '? Unknown', `badge${known ? (cached ? '' : ' missed') : ' unknown'}`));
      row.append(cell);
    }
    body.append(row);
  });
  if (unknownCache) warnings.push('Some article cache decisions are unavailable.');
  if (!pages.length) {
    const row = node('tr');
    const cell = node('td', 'No valid article rows were supplied for this hour.');
    cell.colSpan = 3 + (replay.hasCacheChoices ? data.predictors.length : 0);
    row.append(cell);
    body.append(row);
  }
  element('status').textContent = warnings.length ? `Partial dataset: ${warnings.join(' ')}` : '';
  element('status').hidden = !warnings.length;
  element('replay').hidden = false;
}

function updateControls() {
  if (!replay) return;
  const { hours } = replay;
  const lastIndex = hours.length - 1;
  const playing = playbackTimer !== null;
  element('previous').disabled = selectedIndex === 0;
  element('next').disabled = selectedIndex === lastIndex;
  element('play').disabled = hours.length < 2;
  element('play').textContent = playing ? 'Pause' : selectedIndex === lastIndex && lastIndex > 0 ? 'Replay day' : 'Play';
  element('timeline').max = lastIndex;
  element('timeline').value = selectedIndex;
  element('timeline').disabled = hours.length < 2;
  element('timeline').setAttribute('aria-valuetext', `${dateTime.format(new Date(hours[selectedIndex].hour))} UTC, available hour ${selectedIndex + 1} of ${hours.length}`);
  element('timeline-start').textContent = `${hours[0].hour.slice(11, 16)} UTC`;
  element('timeline-end').textContent = `${hours[lastIndex].hour.slice(11, 16)} UTC`;
  element('playback-state').textContent = hours.length === 1 ? 'Only one hour available' : playing ? 'Playing · 2 seconds per available hour' : selectedIndex === lastIndex ? 'End of replay' : 'Paused · 2 seconds per available hour';
  const gaps = 24 - hours.length;
  element('coverage').textContent = gaps ? `${gaps} hour${gaps === 1 ? '' : 's'} unavailable in this UTC day. The timeline steps through supplied hours only; gaps are not interpolated.` : 'All 24 hourly records are available. Use arrow keys on the timeline to step through the day.';
}

function pause() {
  clearInterval(playbackTimer);
  playbackTimer = null;
  updateControls();
}

function selectHour(index) {
  if (!replay) return;
  pause();
  selectedIndex = Math.max(0, Math.min(replay.hours.length - 1, index));
  render();
  element('replay-announcement').textContent = `Selected ${dateTime.format(new Date(replay.hours[selectedIndex].hour))} UTC.`;
}

function togglePlayback() {
  if (!replay || replay.hours.length < 2) return;
  if (playbackTimer !== null) {
    pause();
    element('replay-announcement').textContent = `Paused at ${dateTime.format(new Date(replay.hours[selectedIndex].hour))} UTC.`;
    return;
  }
  if (selectedIndex === replay.hours.length - 1) selectedIndex = 0;
  playbackTimer = setInterval(() => {
    selectedIndex += 1;
    if (selectedIndex === replay.hours.length - 1) {
      pause();
      element('replay-announcement').textContent = 'End of replay. Playback paused.';
    }
    render();
  }, FRAME_MS);
  render();
  element('replay-announcement').textContent = 'Replay playing. Advances one available hour every two seconds.';
}

async function load() {
  pause();
  approvedRun = null;
  replay = null;
  selectedIndex = 0;
  element('replay').hidden = true;
  element('retry').hidden = true;
  element('status').hidden = false;
  element('status').className = 'status';
  element('status').textContent = 'Loading the supplied replay dataset…';
  element('provenance').textContent = 'DEVELOPMENT DATA / UNVERIFIED';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(SOURCE, { cache: 'no-store', signal: controller.signal });
    if (!response.ok) throw new Error(`The replay file could not be loaded (HTTP ${response.status}).`);
    let data;
    const rawReplay = await response.text();
    try { data = JSON.parse(rawReplay); } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw new Error('The replay file is not valid JSON.');
    }
    replay = { data, ...validate(data) };
    await verifyRun(rawReplay, controller.signal);
    configureExploreReplay();
    await loadEventContext(controller.signal);
    const mainResults = resultDataset(100000, data);
    renderPresentation(mainResults);
    renderForesight(mainResults);
    renderStorageComparison(data);
    initializeSpikes(data);
    render();
    showView(requestedView(), { focus: false, updateUrl: true });
  } catch (error) {
    replay = null;
    element('status').className = 'status error';
    element('replay').hidden = true;
    element('status').textContent = `${error.name === 'AbortError' ? 'Loading timed out.' : error.message} Serve the repository over HTTP and check demo.json. No substitute results have been generated.`;
    element('retry').hidden = false;
  } finally { clearTimeout(timeout); }
}

element('retry').addEventListener('click', load);
element('show-pages').addEventListener('click', () => { showAllPages = !showAllPages; render(); });
element('previous').addEventListener('click', () => selectHour(selectedIndex - 1));
element('next').addEventListener('click', () => selectHour(selectedIndex + 1));
element('timeline').addEventListener('input', event => selectHour(Number(event.target.value)));
element('play').addEventListener('click', togglePlayback);
document.addEventListener('visibilitychange', () => {
  if (document.hidden && playbackTimer !== null) {
    pause();
    element('replay-announcement').textContent = 'Replay paused while the page was hidden.';
  }
});
window.addEventListener('pagehide', pause);
load();
