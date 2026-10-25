'use strict';

let storageMethod = 'our_model';

function renderStorageComparison(data = replay?.data) {
  const container = element('storage-comparison');
  container.replaceChildren();
  const small = resultProfile(10000);
  const large = resultProfile(100000);
  const ready = Boolean(small && large && data && data._fake !== true);
  element('storage-question').textContent = ready
    ? 'We first tested a cache with space for 10,000 articles, then tested 100,000. Here we keep the method and planning time the same to compare the effect of cache capacity.'
    : 'The storage comparison is unavailable until both sets of results match the confirmed run.';
  document.querySelectorAll('[name="storage-method"]').forEach(input => { input.disabled = !ready; });
  element('storage-answer').hidden = !ready;
  if (!ready) return;
  const label = modelLabel(data, storageMethod);
  for (const [index, profile] of [small, large].entries()) {
    const value = profile.summary.hit_rate[storageMethod];
    const card = node('article', undefined, `storage-card${index ? ' larger' : ''}`);
    card.append(node('p', index ? 'More room' : 'Less room', 'eyebrow'), node('h2', `${number.format(profile.cache_size)} articles`));
    const slots = node('div', undefined, 'capacity-picture');
    slots.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 10; i++) slots.append(node('span', undefined, i === 0 || index === 1 ? 'filled' : ''));
    card.append(slots, node('p', label, 'storage-method-label'), node('p', percent.format(value), 'storage-rate'), node('p', 'of recorded views served from fast storage', 'metric-note'));
    const meter = node('div', undefined, 'meter storage-meter');
    meter.setAttribute('role', 'meter');
    meter.setAttribute('aria-label', `${label} with space for ${number.format(profile.cache_size)} articles`);
    meter.setAttribute('aria-valuemin', '0');
    meter.setAttribute('aria-valuemax', '100');
    meter.setAttribute('aria-valuenow', String(value * 100));
    meter.setAttribute('aria-valuetext', percent.format(value));
    const fill = node('div', undefined, 'meter-fill');
    fill.style.width = `${value * 100}%`;
    meter.append(fill);
    card.append(meter, node('p', `About ${Math.round(value * 1000)} out of 1,000 recorded views`, 'storage-equivalent'));
    container.append(card);
  }
  const difference = (large.summary.hit_rate[storageMethod] - small.summary.hit_rate[storageMethod]) * 100;
  element('storage-answer').replaceChildren(node('h2', difference > 0 ? 'More room helped this approach.' : 'Read the measured difference.'),
    node('p', `With ${label.toLowerCase()}, the larger store served ${Math.abs(difference).toFixed(2)} percentage points ${difference >= 0 ? 'more' : 'fewer'} recorded views. Both results use the same 72 evaluated hours and choose articles one hour ahead.`),
    node('p', 'This is the effect of storage capacity. The previous page compared forecasting with the standard rule at the same capacity.', 'metric-note'));
}

document.querySelectorAll('[name="storage-method"]').forEach(input => input.addEventListener('change', event => {
  if (!event.target.checked) return;
  storageMethod = event.target.value;
  renderStorageComparison();
}));
