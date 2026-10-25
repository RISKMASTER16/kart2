'use strict';

// Live cache planner page. Polls the local server (api.py) and renders its /api/live state.
const POLL_MS = 20000;
const number = new Intl.NumberFormat('en-GB');
const pct = value => `${(value * 100).toFixed(2)}%`;
const el = id => document.getElementById(id);
let state = null;

function node(tag, text, className) {
  const n = document.createElement(tag);
  if (text !== undefined && text !== null) n.textContent = text;
  if (className) n.className = className;
  return n;
}

function hourLabel(iso) {
  if (!iso) return '–';
  const d = new Date(iso);
  return `${d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' })}, ${String(d.getUTCHours()).padStart(2, '0')}:00 UTC`;
}

function articleLink(row) {
  const a = node('a', row.title);
  a.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(row.page)}`;
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  return a;
}

function size() {
  return document.querySelector('[name="size"]:checked').value;
}

function renderMode(s) {
  const badge = el('mode');
  badge.dataset.mode = s.mode || 'starting';
  const age = s.latest_age_hours;
  el('eyebrow').textContent = s.mode === 'replay' ? 'REPLAY · ENGLISH WIKIPEDIA' : 'LIVE · ENGLISH WIKIPEDIA';
  el('mode-text').textContent = {
    live: age !== undefined ? `Live · latest data ${age} h old` : 'Live',
    offline: 'Offline · showing the latest data on this computer',
    replay: 'Replay · historical hour shown as if it were now',
  }[s.mode] || 'Starting';
}

function renderScores(s) {
  const n = size();
  const rows = (s.scores || []).filter(r => r[`standard_${n}`] !== undefined);
  const cards = el('score-cards');
  const body = el('scores-body');
  cards.replaceChildren();
  body.replaceChildren();
  if (!rows.length) {
    cards.append(node('p', s.model ? 'Scores appear once an hour has passed after the first prediction.' : 'Scores need the trained XGBoost model (python train_xgb.py).', 'metric-note'));
    return;
  }
  const wins = rows.filter(r => r[`model_${n}`] > r[`standard_${n}`]).length;
  const avg = rows.reduce((t, r) => t + r[`model_${n}`] - r[`standard_${n}`], 0) / rows.length;
  const last = rows[rows.length - 1];
  for (const [label, key, cls] of [['Standard rule', 'standard', 'standard'], ['Our model (XGBoost)', 'model', 'model']]) {
    const card = node('article', undefined, cls);
    card.append(node('h3', label), node('p', `Latest scored hour: ${hourLabel(last.hour)}`, 'metric-note'),
      node('p', pct(last[`${key}_${n}`]), 'rate'), node('p', 'of that hour\'s views served from the cache', 'metric-note'));
    cards.append(card);
  }
  el('scores-note').textContent = `Over the last ${rows.length} scored hours, our model served more views in ${wins} of ${rows.length}, by ${(avg * 100).toFixed(2)} percentage points on average. Each method chose its cache using only data published before the hour.`;
  for (const r of [...rows].reverse()) {
    const tr = node('tr');
    const diff = r[`model_${n}`] - r[`standard_${n}`];
    const d = node('td', `${diff >= 0 ? '+' : ''}${(diff * 100).toFixed(2)} pts`, `num ${diff >= 0 ? 'trend-up' : 'trend-down'}`);
    tr.append(node('td', hourLabel(r.hour)), node('td', pct(r[`standard_${n}`]), 'num'), node('td', pct(r[`model_${n}`]), 'num'), d);
    body.append(tr);
  }
}

function renderPlan(s) {
  const body = el('plan-body');
  body.replaceChildren();
  const cache = s.plan.caches[size()];
  // Pages the model adds are the ones the standard rule would leave out at this cache size.
  const adds = new Set(cache.adds.map(a => a.page));
  s.plan.top.forEach((r, i) => {
    const tr = node('tr');
    const name = node('td');
    name.append(articleLink(r));
    if (adds.has(r.page)) name.append(node('span', 'only our model caches this', 'tag'));
    const change = r.views_now ? r.predicted / r.views_now - 1 : null;
    const changeText = change === null ? 'new' : `${change >= 0 ? '▲' : '▼'} ${Math.abs(change * 100).toFixed(0)}%`;
    tr.append(node('td', String(i + 1)), name, node('td', number.format(r.views_now), 'num'),
      node('td', number.format(r.predicted), 'num'),
      node('td', changeText, `num ${change === null || change >= 0 ? 'trend-up' : 'trend-down'}`));
    body.append(tr);
  });
  el('plan-note').textContent = s.model
    ? `The articles our model expects to be read most between ${hourLabel(s.target_hour)} and the end of that hour.`
    : 'The trained XGBoost model was not found, so this list uses the standard rule. Run python train_xgb.py to enable the model.';
}

function renderDiffs(s) {
  const n = Number(size());
  const cache = s.plan.caches[String(n)];
  el('diff-note').textContent = `With room for ${number.format(n)} articles, the two caches share ${(cache.overlap * 100).toFixed(1)}% of their contents; ${number.format(cache.differ)} articles differ. The largest differences:`;
  for (const [id, list] of [['adds', cache.adds], ['drops', cache.drops]]) {
    const ol = el(id);
    ol.replaceChildren();
    if (!list.length) ol.append(node('li', 'None at this cache size.'));
    for (const r of list) {
      const li = node('li');
      li.append(articleLink(r), node('span', `${number.format(r.views_now)} → ${number.format(r.predicted)} predicted`));
      ol.append(li);
    }
  }
}

function render() {
  if (!state) return;
  renderMode(state);
  el('status').textContent = state.phase === 'ready'
    ? `Updated ${new Date(state.updated_at).toLocaleTimeString('en-GB')}. ${state.hours_available} hours of data loaded${state.history_hours < 25 ? ` (the model uses up to 25; still filling)` : ''}.`
    : state.message;
  if (!state.plan) return;
  el('latest').textContent = hourLabel(state.latest_hour);
  el('target').textContent = hourLabel(state.target_hour);
  el('total').textContent = number.format(state.total_views_latest);
  renderScores(state);
  renderPlan(state);
  renderDiffs(state);
}

async function poll() {
  try {
    const response = await fetch('/api/live', { cache: 'no-store' });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    state = await response.json();
    render();
    setTimeout(poll, state.phase === 'ready' ? POLL_MS : 4000);
  } catch (error) {
    el('mode').dataset.mode = 'offline';
    el('mode-text').textContent = 'Not connected';
    el('status').textContent = `The live planner needs the local server: run "uvicorn api:app --port 8000" from the repository root and open http://127.0.0.1:8000/demo/live.html. (${error.message})`;
    setTimeout(poll, 8000);
  }
}

document.querySelectorAll('[name="size"]').forEach(input => input.addEventListener('change', render));
poll();
