'use strict';

const viewTitles = { start: 'start-title', explore: 'snapshot-title', results: 'results-title', storage: 'storage-title' };
const legacyViews = { 'summary-title': 'results', 'snapshot-title': 'explore', 'spike-title': 'results', 'foresight-title': 'results', 'limits-title': 'results' };

function requestedView() {
  const hash = window.location.hash.slice(1);
  return Object.hasOwn(viewTitles, hash) ? hash : legacyViews[hash] || 'start';
}

function showView(view, { focus = true, updateUrl = true } = {}) {
  if (!Object.hasOwn(viewTitles, view)) view = 'start';
  if (view !== 'start' && !replay) return;
  pause();
  document.querySelectorAll('[data-view]').forEach(section => { section.hidden = section.dataset.view !== view; });
  document.querySelectorAll('.journey-nav [data-go]').forEach(link => {
    if (link.dataset.go === view || (view === 'storage' && link.dataset.go === 'results')) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  if (updateUrl && window.location.hash !== `#${view}`) history.pushState({view}, '', `#${view}`);
  updateContents(view);
  closeContents();
  const heading = element(viewTitles[view]);
  if (focus) {
    window.scrollTo({ top: 0, behavior: 'instant' });
    heading.focus({ preventScroll: true });
  }
}

document.querySelectorAll('[data-go]').forEach(link => link.addEventListener('click', event => {
  event.preventDefault();
  showView(link.dataset.go);
}));
window.addEventListener('popstate', () => showView(requestedView(), { updateUrl: false }));
window.addEventListener('hashchange', () => showView(requestedView(), { updateUrl: false }));


// Build navigation from the existing article headings, without duplicating content.
const sidebar = document.getElementById('article-sidebar');
const contentsToggle = document.getElementById('contents-toggle');
function closeContents() {
  sidebar.classList.remove('is-open');
  contentsToggle.setAttribute('aria-expanded', 'false');
}
function sectionLink(heading, view) {
  const link = document.createElement('a');
  link.href = '#' + heading.id;
  link.textContent = heading.textContent.trim();
  link.addEventListener('click', event => {
    event.preventDefault();
    showView(view);
    let parent = heading.parentElement;
    while (parent && parent !== document.body) {
      if (parent.tagName === 'DETAILS') parent.open = true;
      parent = parent.parentElement;
    }
    heading.setAttribute('tabindex', '-1');
    heading.focus({preventScroll:true});
    heading.scrollIntoView({behavior:'instant', block:'start'});
    document.getElementById('search-results').hidden = true;
  });
  return link;
}
document.querySelectorAll('[data-view]').forEach(section => {
  section.querySelectorAll('h1,h2').forEach((heading, index) => {
    if (!heading.id) heading.id = section.dataset.view + '-section-' + index;
    if (heading.tagName === 'H1') heading.querySelectorAll('br').forEach(br => br.replaceWith(document.createTextNode(' ')));
  });
});
function updateContents(view) {
  const list = document.getElementById('section-links');
  list.replaceChildren();
  document.querySelector('[data-view="' + view + '"]').querySelectorAll('h1,h2').forEach(heading => {
    const item = document.createElement('li');
    item.append(sectionLink(heading, view));
    list.append(item);
  });
}
contentsToggle.addEventListener('click', () => {
  const open = contentsToggle.getAttribute('aria-expanded') !== 'true';
  contentsToggle.setAttribute('aria-expanded', String(open));
  sidebar.classList.toggle('is-open', open);
});
const searchPanel = document.getElementById('search-results');
document.querySelector('.project-search').addEventListener('submit', event => {
  event.preventDefault();
  const query = document.getElementById('project-query').value.trim().toLocaleLowerCase();
  const list = searchPanel.querySelector('ul');
  list.replaceChildren();
  let matches = 0;
  if (query) document.querySelectorAll('[data-view]').forEach(section => {
    section.querySelectorAll('h1,h2').forEach(heading => {
      const context = heading.parentElement;
      if (!context.textContent.toLocaleLowerCase().includes(query)) return;
      if (matches >= 12) return;
      const item = document.createElement('li');
      item.append(sectionLink(heading, section.dataset.view));
      list.append(item);
      matches++;
    });
  });
  document.getElementById('search-message').textContent = !query ? 'Enter a topic to search this project.' : matches ? 'Matching sections in this project' : 'No matching sections. Try cache, model or storage.';
  searchPanel.hidden = false;
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  if (!searchPanel.hidden) { searchPanel.hidden = true; document.getElementById('project-query').focus(); }
  else if (sidebar.classList.contains('is-open')) { closeContents(); contentsToggle.focus(); }
});
document.addEventListener('click', event => {
  if (!event.target.closest('.project-search')) searchPanel.hidden = true;
});
updateContents('start');
