'use strict';

// Read already-scored hourly outputs. No cache selection or backtest runs here.
function csvRows(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && char === ',') { row.push(field); field = ''; }
    else if (!quoted && (char === '\n' || char === '\r')) {
      if (char === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      if (row.some(value => value !== '')) rows.push(row);
      row = []; field = '';
    } else field += char;
  }
  if (quoted) throw new Error('Unclosed CSV field.');
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function configureExploreReplay() {
  replay.capacity = 100000;
  replay.hasCacheChoices = replay.data.cache_size === replay.capacity;
  if (replay.hasCacheChoices) return;
  // Demand is independent of cache size; only the capacity-specific scores change.
  const demandHours = replay.hours;
  replay.hours = demandHours.map(hour => ({ ...hour, hit_rate: {} }));
  if (!approvedRun || replay.data._fake === true || !confirmedHourlyText) return;
  try {
    const rows = csvRows(confirmedHourlyText);
    const headings = rows.shift() || [];
    const fields = Object.fromEntries(['predictor','cache_size','horizon','target_hour','hit_rate'].map(name => [name, headings.indexOf(name)]));
    if (Object.values(fields).some(index => index < 0)) throw new Error('Missing hourly fields.');
    const expected = new Map(replay.data.predictors.map(key => [key === 'our_model' ? replay.data.our_model_name : key, key]));
    const rates = new Map();
    for (const row of rows) {
      if (row[fields.cache_size] !== '100000' || row[fields.horizon] !== '1' || !expected.has(row[fields.predictor])) continue;
      const time = `${row[fields.target_hour]?.replace(' ', 'T')}Z`;
      const supplied = row[fields.hit_rate];
      const value = supplied?.trim() ? Number(supplied) : NaN;
      if (!isHour(time) || !isRate(value)) continue;
      const key = expected.get(row[fields.predictor]);
      if (!rates.has(time)) rates.set(time, {});
      if (Object.hasOwn(rates.get(time), key)) throw new Error('Duplicate hourly result.');
      rates.get(time)[key] = value;
    }
    replay.hours = demandHours.map(hour => ({ ...hour, hit_rate: rates.get(hour.hour) || {} }));
  } catch {
    // Keep demand visible but never substitute the 10k cache results.
    replay.hours = demandHours.map(hour => ({ ...hour, hit_rate: {} }));
  }
}
