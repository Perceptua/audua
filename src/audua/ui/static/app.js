/* audua UI — vanilla, no build step.
 *
 * Four views over one read-only API, plus two panes that outlive navigation:
 * the reading pane (markdown, rendered server-side) and the player (one
 * <audio> element that keeps playing while you browse). Every link inside a
 * rendered document that points at a clip is rewritten by the server into a
 * data-audio / data-doc action, so a citation opens here instead of leaving.
 */

const view = document.getElementById('view');
const workspace = document.getElementById('workspace');
const gutter = document.getElementById('gutter');
const reader = document.getElementById('reader');
const readerBody = document.getElementById('reader-body');
const readerTitle = document.getElementById('reader-title');
const readerSub = document.getElementById('reader-sub');
const railLabel = document.getElementById('rail-label');
const collapseButton = document.getElementById('reader-collapse');
const player = document.getElementById('player');
const audio = document.getElementById('audio');
const playerTitle = document.getElementById('player-title');
const playerSub = document.getElementById('player-sub');

/* The run whose documents and clips the panes are currently showing. */
let currentRun = null;
let playing = null;

/* ---------------------------------------------------------------- utils */

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function bytes(n) {
  if (!n) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

function ago(iso) {
  if (!iso) return '—';
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return iso;
  const secs = (Date.now() - then.getTime()) / 1000;
  if (secs < 90) return 'just now';
  const steps = [[60, 'minute'], [60, 'hour'], [24, 'day'], [7, 'week']];
  let value = secs;
  for (const [size, name] of steps) {
    value /= size;
    if (value < (name === 'day' ? 7 : 60) || name === 'week') {
      if (value < 1) continue;
      return `${Math.round(value)} ${name}${Math.round(value) === 1 ? '' : 's'} ago`;
    }
  }
  return then.toLocaleDateString();
}

function when(iso) {
  if (!iso) return '—';
  const at = new Date(iso);
  return Number.isNaN(at.getTime()) ? iso : at.toLocaleString();
}

function clock(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  const payload = await response.json().catch(() => ({ error: response.statusText }));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function pill(status) {
  const label = { ok: 'ok', ready: 'ready', processed: 'processed', failed: 'failed',
                  incomplete: 'incomplete' }[status] || status;
  return `<span class="pill ${esc(status)}">${esc(label)}</span>`;
}

function flags(list, extraClass = '') {
  if (!list || !list.length) return '';
  return list.map((f) => `<span class="flag ${extraClass}">${esc(f)}</span>`).join('');
}

function empty(title, detail) {
  return `<div class="empty"><strong>${esc(title)}</strong>${esc(detail || '')}</div>`;
}

/* ----------------------------------------------------------- split pane */

/* The width the user asked for, which is not always the width they get: a
 * narrow window clamps it, and widening the window again should give it back
 * rather than leaving the pane where the clamp pinned it. */
const WIDTH_KEY = 'audua.readerWidth';
const COLLAPSED_KEY = 'audua.readerCollapsed';
const MIN_READER = 300;
const MIN_MAIN = 380;
const DEFAULT_FRACTION = 0.46;

/* Layout preferences are a nicety; a browser that refuses storage should get a
 * working split pane that simply forgets it between visits. */
const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* nothing to do */ }
  },
};

/* CSS clamps what is *displayed* (see .reader). This clamps what is *stored*,
 * and only ever runs while the user is driving the divider — so a preference
 * is never quietly rewritten by a window that happened to be narrow. */
function setReaderWidth(width) {
  const ceiling = Math.max(MIN_READER, window.innerWidth - MIN_MAIN);
  const chosen = Math.round(Math.min(Math.max(width, MIN_READER), ceiling));
  workspace.style.setProperty('--reader-width', `${chosen}px`);
  store.set(WIDTH_KEY, String(chosen));
  return chosen;
}

/* Nudges work off the width on screen, not the stored one, so an arrow key
 * moves the divider you can see. */
const shownWidth = () => reader.getBoundingClientRect().width;

function resetReaderWidth() {
  setReaderWidth(window.innerWidth * DEFAULT_FRACTION);
}

function setCollapsed(collapsed) {
  document.body.classList.toggle('reader-collapsed', collapsed);
  collapseButton.innerHTML = collapsed ? '&#171;' : '&#187;';
  collapseButton.title = collapsed ? 'Expand the reading pane' : 'Collapse the reading pane';
  collapseButton.setAttribute('aria-expanded', String(!collapsed));
  store.set(COLLAPSED_KEY, collapsed ? '1' : '0');
}

const isCollapsed = () => document.body.classList.contains('reader-collapsed');

/* Dragging. Pointer events cover mouse, pen and touch in one path. Capture
 * keeps the drag alive when the pointer outruns the 9px gutter, but the move
 * and release handlers sit on the window so the drag still works if capture
 * is refused — a pointer can go inactive between down and capture. */
let dragPointer = null;

gutter.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  dragPointer = event.pointerId;
  try {
    gutter.setPointerCapture(event.pointerId);
  } catch { /* capture is an optimisation, not the mechanism */ }
  document.body.classList.add('resizing');
});

window.addEventListener('pointermove', (event) => {
  if (event.pointerId !== dragPointer) return;
  setReaderWidth(window.innerWidth - event.clientX);
});

const endDrag = (event) => {
  if (event.pointerId !== dragPointer) return;
  try {
    gutter.releasePointerCapture(event.pointerId);
  } catch { /* never captured, or already released */ }
  dragPointer = null;
  document.body.classList.remove('resizing');
};
window.addEventListener('pointerup', endDrag);
window.addEventListener('pointercancel', endDrag);
gutter.addEventListener('dblclick', resetReaderWidth);

gutter.addEventListener('keydown', (event) => {
  const step = event.shiftKey ? 64 : 16;
  if (event.key === 'ArrowLeft') setReaderWidth(shownWidth() + step);
  else if (event.key === 'ArrowRight') setReaderWidth(shownWidth() - step);
  else if (event.key === 'Home') resetReaderWidth();
  else if (event.key === 'Enter' || event.key === ' ') setCollapsed(true);
  else return;
  event.preventDefault();
});

collapseButton.addEventListener('click', () => setCollapsed(!isCollapsed()));
document.getElementById('reader-rail').addEventListener('click', () => setCollapsed(false));

/* Restore the stored preference verbatim; CSS decides how much of it fits. */
const stored = Number(store.get(WIDTH_KEY));
workspace.style.setProperty(
  '--reader-width',
  `${Math.round(stored >= MIN_READER ? stored : window.innerWidth * DEFAULT_FRACTION)}px`,
);
setCollapsed(store.get(COLLAPSED_KEY) === '1');

/* --------------------------------------------------------------- reader */

function closeReader() {
  reader.hidden = true;
  document.body.classList.remove('reader-open');
}

/* `expand` is false for the summary a detail view opens by itself: arriving at
 * a run should not re-open a pane the reader deliberately collapsed. Clicking
 * a citation or a transcript is a request to *see* it, so that one expands. */
async function openDoc(run, file, subtitle, { expand = true } = {}) {
  currentRun = run;
  reader.hidden = false;
  document.body.classList.add('reader-open');
  if (expand) setCollapsed(false);
  readerTitle.textContent = file;
  readerSub.textContent = subtitle || run;
  railLabel.textContent = file;
  readerBody.innerHTML = '<div class="loading">Reading…</div>';
  try {
    const doc = await api(`/api/outputs/${encodeURIComponent(run)}/doc?file=${encodeURIComponent(file)}`);
    readerBody.innerHTML = doc.html;
    readerSub.textContent = `${run} · ${bytes(doc.size)} · ${ago(doc.modified)}`;
    readerBody.scrollTop = 0;
  } catch (error) {
    readerBody.innerHTML = `<p class="muted">Could not open ${esc(file)}: ${esc(error.message)}</p>`;
  }
}

/* --------------------------------------------------------------- player */

function playClip(run, file, title, subtitle) {
  currentRun = run;
  const source = `/media/${encodeURIComponent(run)}/${encodeURIComponent(file)}`;
  if (audio.getAttribute('src') !== source) {
    audio.src = source;
  }
  player.hidden = false;
  playerTitle.textContent = title || file;
  playerSub.textContent = subtitle || run;
  playing = file;
  markPlaying();
  audio.play().catch(() => { /* the user can press play; autoplay may be blocked */ });
}

function markPlaying() {
  document.querySelectorAll('.play').forEach((button) => {
    button.classList.toggle('playing', button.dataset.audio === playing && !audio.paused);
    button.innerHTML = button.classList.contains('playing') ? '&#9646;&#9646;' : '&#9654;';
  });
}

audio.addEventListener('play', markPlaying);
audio.addEventListener('pause', markPlaying);
audio.addEventListener('ended', markPlaying);

document.getElementById('player-close').addEventListener('click', () => {
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  player.hidden = true;
  playing = null;
  markPlaying();
});
document.getElementById('reader-close').addEventListener('click', closeReader);

/* Clicks on anything the server rendered, anywhere in the page. */
document.addEventListener('click', (event) => {
  const audioLink = event.target.closest('[data-audio]');
  if (audioLink) {
    event.preventDefault();
    const run = audioLink.dataset.run || currentRun;
    if (run) {
      if (playing === audioLink.dataset.audio && !audio.paused) audio.pause();
      else playClip(run, audioLink.dataset.audio, audioLink.dataset.label, audioLink.dataset.sub);
    }
    return;
  }

  const docLink = event.target.closest('[data-doc]');
  if (docLink) {
    event.preventDefault();
    const run = docLink.dataset.run || currentRun;
    if (run) openDoc(run, docLink.dataset.doc, docLink.dataset.sub);
    return;
  }

  /* Footnote jumps stay inside the pane instead of driving the router. */
  const anchor = event.target.closest('.doc a[href^="#"]');
  if (anchor) {
    event.preventDefault();
    const target = readerBody.querySelector(`[id="${CSS.escape(anchor.getAttribute('href').slice(1))}"]`);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.classList.add('flash');
      setTimeout(() => target.classList.remove('flash'), 1600);
    }
  }
});

/* ------------------------------------------------------------ dashboard */

function renderDashboard(data) {
  const { inbox, outputs, clips, latest_batch: batch, roots } = data;

  const stats = [
    { n: inbox.ready, k: 'waiting', x: inbox.ready ? bytes(inbox.ready_bytes) : 'inbox empty',
      href: '#/processing', cls: inbox.ready ? 'is-ready' : '' },
    { n: inbox.processed, k: 'processed', x: 'filed under processed/', href: '#/processing',
      cls: 'is-ok' },
    { n: inbox.failed, k: 'failed', x: 'filed under failed/', href: '#/processing',
      cls: inbox.failed ? 'is-bad' : '' },
    { n: outputs.total, k: 'runs', x: `${outputs.summarized} summarized`, href: '#/outputs' },
    { n: outputs.awaiting_summary, k: 'awaiting summary',
      x: outputs.awaiting_summary ? 'run the audua-summarize skill' : 'all caught up',
      href: '#/outputs', cls: outputs.awaiting_summary ? 'is-ready' : 'is-ok' },
    { n: clips.total, k: 'clips', x: `${clips.speech_hms} of speech · ${clips.flagged} flagged`,
      href: '#/outputs' },
  ];

  const statHtml = stats.map((s) => `
    <a class="card stat ${s.cls || ''}" href="${s.href}">
      <div class="n">${esc(s.n)}</div>
      <div class="k">${esc(s.k)}</div>
      <div class="x">${esc(s.x)}</div>
    </a>`).join('');

  let batchHtml = `<div class="card panel"><h3>Last batch</h3>
    <p class="muted">No batch has been run against this tree yet.
    Run <code>uv run audua batch</code> to work the inbox.</p></div>`;

  if (batch) {
    const counts = batch.counts || {};
    const rows = (batch.results || []).map((r) => `
      <tr>
        <td>${esc(r.source)}</td>
        <td>${pill(r.status)}</td>
        <td class="num">${r.clip_count == null ? '—' : esc(r.clip_count) + ' clips'}</td>
        <td>${r.reasons.length ? `<span class="reasons">${esc(r.reasons.join('; '))}</span>` : ''}</td>
      </tr>`).join('');

    batchHtml = `<div class="card panel">
      <h3>Last batch — ${esc(ago(batch.started))}</h3>
      <div class="kv">
        <span><b>${esc(counts.processed || 0)}</b> processed</span>
        <span><b>${esc(counts.failed || 0)}</b> failed</span>
        <span><b>${esc(counts.skipped || 0)}</b> skipped</span>
        <span><b>${esc(counts.summaries_pending || 0)}</b> summaries pending</span>
        <span>took ${esc(clock(batch.elapsed_seconds))}</span>
      </div>
      ${batch.aborted ? `<p class="reasons">Aborted: ${esc(batch.aborted)}</p>` : ''}
      ${rows ? `<div class="wrap" style="margin-top:12px"><table class="grid"><tbody>${rows}</tbody></table></div>` : ''}
      ${(batch.skipped || []).length ? `<p class="muted" style="margin-bottom:0">Skipped: ${
        esc(batch.skipped.map((s) => `${s.source} (${s.reason})`).join('; '))}</p>` : ''}
    </div>`;
  }

  const waitingHtml = data.waiting.length ? `
    <h2 class="section">Waiting in the inbox</h2>
    <div class="card wrap"><table class="grid"><tbody>
      ${data.waiting.map((s) => `<tr>
        <td>${esc(s.name)}${s.has_overrides ? ' <span class="flag muted">overrides</span>' : ''}</td>
        <td class="num">${esc(bytes(s.size))}</td>
        <td class="num">${esc(ago(s.modified))}</td>
      </tr>`).join('')}
    </tbody></table></div>` : '';

  const recentHtml = data.recent_runs.length ? `
    <h2 class="section">Recent output</h2>
    <div class="runs">${data.recent_runs.map(runCard).join('')}</div>` : '';

  view.innerHTML = `
    <div class="page-head">
      <h1>Overview</h1>
      <div class="sub">inbox <code>${esc(roots.raw)}</code> · outputs <code>${esc(roots.output)}</code></div>
    </div>
    <div class="stats">${statHtml}</div>
    ${batchHtml}
    ${waitingHtml}
    ${recentHtml}`;
}

/* ----------------------------------------------------------- processing */

function sourceTable(rows) {
  return `<div class="card wrap"><table class="grid">
    <thead><tr>
      <th>Recording</th><th>Status</th><th>Size</th><th>Length</th>
      <th>Clips</th><th>Last run</th><th>Output</th>
    </tr></thead>
    <tbody>${rows.map((s) => `
      <tr>
        <td>
          ${esc(s.name)}
          ${s.has_overrides ? '<span class="flag muted">overrides</span>' : ''}
          ${s.reasons.length ? `<div class="reasons">${esc(s.reasons.join('; '))}</div>` : ''}
        </td>
        <td>${pill(s.status)}</td>
        <td class="num">${esc(bytes(s.size))}</td>
        <td class="num">${esc(s.duration_hms || '—')}</td>
        <td class="num">${s.clip_count == null ? '—' : `${esc(s.clip_count)}${
          s.flagged_clips ? ` <span class="flag">${esc(s.flagged_clips)} flagged</span>` : ''}`}</td>
        <td class="num">${esc(s.last_run ? ago(s.last_run) : ago(s.modified))}</td>
        <td>${s.run ? `<a href="#/outputs/${encodeURIComponent(s.run)}">${esc(s.run)}</a>`
                    : '<span class="muted">—</span>'}</td>
      </tr>`).join('')}
    </tbody></table></div>`;
}

function renderProcessing(data) {
  const sources = data.sources;
  const groups = [
    ['ready', 'Ready — waiting in the inbox'],
    ['failed', 'Failed'],
    ['processed', 'Processed'],
  ];

  const sections = groups.map(([status, title]) => {
    const rows = sources.filter((s) => s.status === status);
    if (!rows.length) return '';
    return `<h2 class="section">${esc(title)} · ${rows.length}</h2>${sourceTable(rows)}`;
  }).join('');

  view.innerHTML = `
    <div class="page-head">
      <h1>Processing</h1>
      <div class="sub">${esc(plural(sources.length, 'recording'))} in the tree.
        A recording is <em>ready</em> while it is still sitting in the inbox;
        the batch files it under processed/ or failed/ when it is done.</div>
    </div>
    ${sections || empty('Nothing here yet', 'Drop a recording into the inbox and run audua batch.')}`;
}

/* -------------------------------------------------------------- outputs */

function runCard(run) {
  const badges = [pill(run.status)];
  if (!run.has_summary && run.status === 'ok') badges.push('<span class="pill pending">summary pending</span>');
  if (run.flagged_clips) badges.push(`<span class="flag">${esc(run.flagged_clips)} flagged</span>`);

  const meta = [
    run.duration_hms,
    `${run.clip_count} clips`,
    `${run.speech_hms} speech`,
    run.model,
  ].filter(Boolean).join(' · ');

  const line = run.summary_line
    ? `<div class="run-line">${esc(run.summary_line)}</div>`
    : `<div class="run-line pending">${run.status === 'failed'
        ? esc(run.reasons.join('; '))
        : 'No summary written yet — the transcript digest is ready for one.'}</div>`;

  return `<a class="card run" href="#/outputs/${encodeURIComponent(run.name)}">
    <div class="run-top">
      <span class="run-name">${esc(run.name)}</span>
      ${badges.join(' ')}
      <span class="run-meta">${esc(meta)}</span>
    </div>
    ${line}
  </a>`;
}

function renderOutputs(data) {
  const runs = data.outputs;
  view.innerHTML = `
    <div class="page-head">
      <h1>Outputs</h1>
      <div class="sub">${esc(plural(runs.length, 'run'))} ·
        ${esc(runs.filter((r) => r.has_summary).length)} summarized</div>
    </div>
    ${runs.length
      ? `<div class="runs">${runs.map(runCard).join('')}</div>`
      : empty('No output yet', 'Run audua batch to process the inbox.')}`;
}

/* --------------------------------------------------------- output detail */

function checkRow(state, label, problems) {
  const list = (problems || []).length
    ? `<ul>${problems.map((p) => `<li>${esc(p)}</li>`).join('')}</ul>` : '';
  return `<div class="check ${state}"><span class="dot"></span><div>${label}${list}</div></div>`;
}

function renderDetail(run) {
  currentRun = run.name;

  const facts = [
    run.source_name && `source <code>${esc(run.source_name)}</code>`,
    run.duration_hms && `${esc(run.duration_hms)} of audio`,
    `${esc(run.clip_count)} clips`,
    `${esc(run.speech_hms)} of speech (${(run.retained_fraction * 100).toFixed(1)}%)`,
    run.model && `faster-whisper <code>${esc(run.model)}</code>`,
    run.completed && `finished ${esc(ago(run.completed))}`,
  ].filter(Boolean).map((f) => `<span>${f}</span>`).join('');

  const documents = [
    ['summary.md', run.has_summary, 'the written summary, with citations'],
    ['transcript_digest.md', run.has_digest, 'every clip, timestamped, in one file'],
    ['manifest.json', run.has_manifest, 'what the pipeline did'],
  ].map(([file, present, hint]) => (present
    ? `<button class="button ${file === 'summary.md' ? 'primary' : ''}" data-doc="${esc(file)}"
        data-run="${esc(run.name)}" title="${esc(hint)}">${esc(file)}</button>`
    : `<button class="button" disabled title="not written yet">${esc(file)}</button>`)).join('');

  const checks = [];
  if (run.status === 'failed') {
    checks.push(checkRow('bad', '<b>This run is marked failed.</b>', run.reasons));
  } else if (run.status === 'incomplete') {
    checks.push(checkRow('info', '<b>Incomplete.</b>', run.reasons));
  }
  if (run.pairing) {
    checks.push(run.pairing.ok
      ? checkRow('ok', 'Clip / transcript pairing intact, both directions.')
      : checkRow('bad', 'Pairing problems on disk:', run.pairing.problems));
  }
  if (run.citations && run.citations.present) {
    checks.push(run.citations.ok
      ? checkRow('ok', `Summary citations resolve — ${esc(plural(run.citations.cited.length, 'clip'))} cited.`)
      : checkRow('bad', 'Citation problems in summary.md:', run.citations.problems));
    if (run.citations.uncited.length) {
      checks.push(checkRow('info',
        `${esc(plural(run.citations.uncited.length, 'clip'))} nothing cites: ` +
        `<span class="muted">${esc(run.citations.uncited.join(', '))}</span>`));
    }
  } else if (run.status === 'ok') {
    checks.push(checkRow('info', 'No summary.md yet — nothing to check citations against.'));
  }
  const flagList = Object.entries(run.flag_counts || {});
  if (flagList.length) {
    checks.push(checkRow('info', `Flagged clips: ${flagList
      .map(([flag, n]) => `<span class="flag">${esc(flag)} ${esc(n)}</span>`).join(' ')}
      <span class="muted">— flags are a record, not a failure.</span>`));
  }

  const clipRows = (run.clips || []).map((clip) => {
    if (clip.status === 'failed') {
      return `<tr class="clip failed">
        <td></td><td class="num">${esc(clip.index)}</td>
        <td class="num">${esc(clip.start_hms)}</td><td class="num">—</td>
        <td colspan="2">failed: ${esc(clip.error || 'unknown error')}</td></tr>`;
    }
    return `<tr class="clip ${clip.cited ? '' : 'uncited'}">
      <td><button class="play" data-audio="${esc(clip.audio_file)}" data-run="${esc(run.name)}"
        data-label="${esc(clip.stem)}"
        data-sub="${esc(run.name)} · ${esc(clip.start_hms)}–${esc(clip.end_hms)}"
        title="Play ${esc(clip.stem)}">&#9654;</button></td>
      <td class="num">${esc(clip.index)}</td>
      <td class="num">${esc(clip.start_hms)}</td>
      <td class="num">${esc(clock(clip.duration))}</td>
      <td>
        <div class="clip-text ${clip.empty ? 'empty' : ''}">${
          clip.empty ? 'no speech transcribed' : esc(clip.preview)}</div>
        <div>${flags(clip.flags)}${clip.cited ? '' : '<span class="flag muted">uncited</span>'}</div>
      </td>
      <td>${clip.text_file
        ? `<button class="linkish" data-doc="${esc(clip.text_file)}" data-run="${esc(run.name)}"
            data-sub="${esc(clip.stem)}">transcript</button>` : ''}</td>
    </tr>`;
  }).join('');

  view.innerHTML = `
    <div class="page-head">
      <div class="sub"><a href="#/outputs">← Outputs</a></div>
      <div class="detail-head">
        <h1>${esc(run.name)}</h1>
        ${pill(run.status)}
      </div>
      <div class="facts">${facts}</div>
    </div>

    <div class="actions">${documents}</div>
    <div class="checks">${checks.join('')}</div>

    <h2 class="section">Clips · ${esc(run.clips ? run.clips.length : 0)}</h2>
    ${clipRows ? `<div class="card wrap"><table class="grid">
      <thead><tr><th></th><th>#</th><th>At</th><th>Length</th><th>Transcript</th><th></th></tr></thead>
      <tbody>${clipRows}</tbody></table></div>`
      : empty('No clips', run.has_manifest
          ? 'The VAD heard no speech in this recording.'
          : 'Nothing has been written here yet — no manifest.json to read.')}`;

  if (run.has_summary) openDoc(run.name, 'summary.md', null, { expand: false });
  else if (run.has_digest) openDoc(run.name, 'transcript_digest.md', null, { expand: false });
  else closeReader();
}

/* --------------------------------------------------------------- router */

const routes = [
  { match: /^$/, load: () => api('/api/overview').then(renderDashboard) },
  { match: /^processing$/, load: () => api('/api/sources').then(renderProcessing) },
  { match: /^outputs$/, load: () => api('/api/outputs').then(renderOutputs) },
  {
    match: /^outputs\/(.+)$/,
    load: (name) => api(`/api/outputs/${encodeURIComponent(decodeURIComponent(name))}`).then(renderDetail),
  },
];

async function route() {
  const path = location.hash.replace(/^#\/?/, '').replace(/\/$/, '');

  document.querySelectorAll('#nav a').forEach((link) => {
    const target = link.dataset.route;
    link.classList.toggle('active', target === path || (target === 'outputs' && path.startsWith('outputs')));
  });

  if (!path.startsWith('outputs/')) closeReader();

  for (const { match, load } of routes) {
    const found = path.match(match);
    if (!found) continue;
    view.innerHTML = '<div class="loading">Reading the filetree…</div>';
    try {
      await load(found[1]);
    } catch (error) {
      view.innerHTML = `<div class="page-head"><h1>Not available</h1></div>
        <div class="card panel"><p class="reasons">${esc(error.message)}</p>
        <p class="muted">The tree is read live, so this usually means the folder moved,
        or the file is being rewritten right now. Try refreshing.</p></div>`;
    }
    return;
  }
  location.hash = '#/';
}

window.addEventListener('hashchange', route);
document.getElementById('refresh').addEventListener('click', route);
route();
