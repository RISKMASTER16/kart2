'use strict';

// Approval belongs to a specific published bundle, never merely to an absent _fake flag.
let approvedRun = null;
let provenanceMessage = 'Run confirmation unavailable.';
let confirmedHourlyText = null;

async function verifyRun(rawReplay, signal) {
  approvedRun = null;
  confirmedHourlyText = null;
  try {
    const response = await fetch('run-metadata.json', { cache: 'no-store', signal });
    if (!response.ok) throw new Error('Run confirmation file unavailable.');
    const manifest = await response.json();
    const paths = [SOURCE, '../out/results.csv', '../out/hourly.csv', '../out/charts/key_numbers.md', '../out/charts_100k/key_numbers.md'];
    if (manifest.status !== 'team-confirmed' || manifest.replay_path !== SOURCE
      || !Array.isArray(manifest.files) || manifest.files.length !== paths.length
      || new Set(manifest.files.map(file => file.path)).size !== paths.length
      || !manifest.files.every(file => paths.includes(file.path) && /^[a-f0-9]{64}$/.test(file.sha256))
      || ![1, 2, 3, 6].every(h => isCount(manifest.hours_by_horizon?.[h]) && manifest.hours_by_horizon[h] > 0)
      || !Array.isArray(manifest.paired_comparisons) || manifest.paired_comparisons.length !== 2
      || !manifest.paired_comparisons.every(item => [1, 6].includes(item.horizon) && isCount(item.n_hours)
        && [item.difference_pp, item.ci_low_pp, item.ci_high_pp, item.better_hours_percent].every(Number.isFinite))
      || !isRate(manifest.spike_hit_rate?.last_hour) || !isRate(manifest.spike_hit_rate?.our_model)
      || ![10000, 100000].every(size => {
        const profile = manifest.capacities?.[size];
        return profile?.cache_size === size && isRate(profile.summary?.oracle_hit_rate)
          && ['last_hour','our_model'].every(key => isRate(profile.summary?.hit_rate?.[key])
            && Array.isArray(profile.summary?.foresight_curve?.hit_rate?.[key])
            && profile.summary.foresight_curve.hit_rate[key].length === 4
            && profile.summary.foresight_curve.hit_rate[key].every(isRate))
          && Array.isArray(profile.paired_comparisons) && profile.paired_comparisons.length === 2;
      })) {
      throw new Error('Run confirmation does not match this data source.');
    }
    let hourlyText = null;
    for (const file of manifest.files) {
      let text;
      if (file.path === SOURCE) text = rawReplay;
      else {
        const result = await fetch(file.path, { cache: 'no-store', signal });
        if (!result.ok) throw new Error('A confirmed result file is unavailable.');
        text = await result.text();
      }
      // Git may check out CRLF on Windows and LF elsewhere. Canonicalise only EOLs.
      const bytes = new TextEncoder().encode(text.replace(/\r\n/g, '\n'));
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const hash = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
      if (hash !== file.sha256) throw new Error('Results changed since this run was confirmed.');
      if (file.path === '../out/hourly.csv') hourlyText = text;
    }
    approvedRun = manifest;
    confirmedHourlyText = hourlyText;
    provenanceMessage = `Published run confirmed by the team. Revision ${manifest.revision}. Evaluated from ${manifest.test_start} to ${manifest.test_end} UTC. Confirmation is recorded for these exact files.`;
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    provenanceMessage = `${error.message} Results remain unverified.`;
  }
}

function provenanceLabel(data) {
  return data._fake === true ? 'DEMO DATA · SYNTHETIC' : approvedRun ? 'MEASURED DATA · CONFIRMED BY TEAM' : 'DEVELOPMENT DATA · NOT VERIFIED';
}

function resultProfile(capacity) {
  return replay?.data._fake === true ? null : approvedRun?.capacities?.[capacity] || null;
}

function resultDataset(capacity, data) {
  return { ...data, cache_size: capacity, summary: resultProfile(capacity)?.summary || {} };
}
