// CSITLens frontend — talks to the FastAPI backend in api.py.
// Served same-origin (mounted via StaticFiles), so relative paths work.

const TONE_BY_LABEL = {
  'Very High': 'tone-safe',
  'High': 'tone-gold',
  'Moderate': 'tone-moderate',
  'Low': 'tone-reach',
  'Very Low': 'tone-danger',
  'Not Found': 'tone-unknown',
};

let COLLEGES = [];       // from GET /colleges
let selectedNames = [];  // canonical college names the student picked, in priority order
let evaluating = false;

function collegeLoc(fullName) {
  const idx = fullName.indexOf(',');
  return idx === -1 ? '' : fullName.slice(idx + 1).trim();
}

function collegeShortName(fullName) {
  const idx = fullName.indexOf(',');
  return idx === -1 ? fullName : fullName.slice(0, idx).trim();
}

function pickCutoff(obj) {
  if (obj.Predicted_Cutoff_Rank != null) return obj.Predicted_Cutoff_Rank;
  if (obj.Latest_Cutoff_Rank != null) return obj.Latest_Cutoff_Rank;
  if (obj.Avg_Cutoff_Rank_AllYears != null) return obj.Avg_Cutoff_Rank_AllYears;
  return null;
}

function toneFor(label) {
  return TONE_BY_LABEL[label] || 'tone-unknown';
}

async function loadColleges() {
  const select = document.getElementById('collegeSelect');
  const addBtn = document.getElementById('addBtn');
  const basisNote = document.getElementById('basisNote');
  try {
    const res = await fetch('/colleges');
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    COLLEGES = await res.json();
    basisNote.textContent = `Cutoff basis: ${COLLEGES.length} colleges on file · admission classifier`;
    renderCollegeSelect();
    renderSelectedList();
  } catch (err) {
    select.innerHTML = '<option value="">Failed to load colleges</option>';
    select.disabled = true;
    addBtn.disabled = true;
    basisNote.textContent = '';
    document.getElementById('selectedList').innerHTML =
      `<div class="error-note">Could not reach the server to load the college list (${err.message}). Make sure the API is running and reload the page.</div>`;
  }
}

function adjustRank(delta) {
  const input = document.getElementById('rankInput');
  input.value = Math.max(1, parseInt(input.value || 0, 10) + delta);
}

function addCollege() {
  const sel = document.getElementById('collegeSelect');
  const name = sel.value;
  if (name && !selectedNames.includes(name)) {
    selectedNames.push(name);
    renderSelectedList();
    renderCollegeSelect();
  }
}

function removeCollege(name) {
  selectedNames = selectedNames.filter(x => x !== name);
  renderSelectedList();
  renderCollegeSelect();
}

function moveCollege(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= selectedNames.length) return;
  const arr = selectedNames;
  [arr[index], arr[target]] = [arr[target], arr[index]];
  renderSelectedList();
}

function renderSelectedList() {
  const container = document.getElementById('selectedList');
  const hint = document.getElementById('countHint');
  hint.textContent = selectedNames.length + ' added';
  if (selectedNames.length === 0) {
    container.innerHTML = '<div class="empty-note">You haven\'t added any colleges yet — add one below to see how your rank compares.</div>';
    return;
  }
  container.innerHTML = selectedNames.map((name, i) => `
    <div class="selected-row">
      <div class="rank-badge">${i + 1}</div>
      <div class="info">
        <div class="name">${collegeShortName(name)}</div>
        <div class="loc">${collegeLoc(name)}</div>
      </div>
      <div class="reorder-btns">
        <button class="order-btn" onclick="moveCollege(${i}, -1)" ${i === 0 ? 'disabled' : ''} aria-label="Move up">▲</button>
        <button class="order-btn" onclick="moveCollege(${i}, 1)" ${i === selectedNames.length - 1 ? 'disabled' : ''} aria-label="Move down">▼</button>
      </div>
      <button class="remove-btn" onclick="removeCollege('${name.replace(/'/g, "\\'")}')" aria-label="Remove">×</button>
    </div>`).join('');
}

function renderCollegeSelect() {
  const sel = document.getElementById('collegeSelect');
  const addBtn = document.getElementById('addBtn');
  const available = COLLEGES.filter(c => !selectedNames.includes(c.College));
  if (available.length === 0) {
    sel.innerHTML = '<option value="">All colleges added</option>';
    sel.disabled = true;
    addBtn.disabled = true;
    return;
  }
  sel.disabled = false;
  addBtn.disabled = false;
  sel.innerHTML = '<option value="">Select a college to add…</option>' +
    available.map(c => `<option value="${c.College}">${collegeShortName(c.College)} — ${collegeLoc(c.College)}</option>`).join('');
}

function renderBestCallout(best) {
  const container = document.getElementById('bestCallout');
  if (!best) {
    container.innerHTML = '';
    return;
  }
  const pct = Math.round(best.Admission_Probability * 100);
  container.innerHTML = `
    <div class="best-callout">
      <span class="tag">Best from your list</span>
      <div class="info">
        <p class="college">${collegeShortName(best.Matched_College)}</p>
        <p class="detail">${collegeLoc(best.Matched_College)} · ${pct}% · ${best.Probability_Label}</p>
      </div>
    </div>`;
}

function renderTicketList(preferences) {
  const container = document.getElementById('ticketList');
  if (!preferences || preferences.length === 0) {
    container.innerHTML = '<div class="empty-note">Add colleges above and select "Evaluate My Chances" to see them scored here.</div>';
    return;
  }
  // API sorts by probability; restore the student's own priority order for this section.
  const ordered = [...preferences].sort((a, b) => a.Preference_Order - b.Preference_Order);

  container.innerHTML = ordered.map(p => {
    if (!p.Matched_College) {
      return `<div class="ticket">
        <div class="stub"><div class="num">—</div><div class="lbl">cutoff</div></div>
        <div class="body">
          <div class="info">
            <p class="college">${p.User_Input}</p>
            <p class="loc">No confident match found for this college name</p>
          </div>
          <div class="side">
            <span class="gapval">—</span>
            <span class="badge tone-unknown">Not Found</span>
          </div>
        </div>
      </div>`;
    }
    const cutoff = p.Destination_Cutoff_Rank != null ? Math.round(p.Destination_Cutoff_Rank) : '—';
    const gap = p.Rank_Gap != null ? Math.round(p.Rank_Gap) : null;
    const gapDisplay = gap == null ? '—' : (gap >= 0 ? `+${gap}` : `${gap}`);
    const pct = p.Admission_Probability != null ? Math.round(p.Admission_Probability * 100) : null;
    const badgeText = pct != null ? `${pct}% · ${p.Probability_Label}` : p.Probability_Label;
    return `<div class="ticket">
      <div class="stub"><div class="num">${cutoff}</div><div class="lbl">cutoff</div></div>
      <div class="body">
        <div class="info">
          <p class="college">${collegeShortName(p.Matched_College)}</p>
          <p class="loc">${collegeLoc(p.Matched_College)}</p>
        </div>
        <div class="side">
          <span class="gapval">${gapDisplay}</span>
          <span class="badge ${toneFor(p.Probability_Label)}">${badgeText}</span>
        </div>
      </div>
    </div>`;
  }).join('');
}

function renderRecommendations(top5, sourceCollege, top5Note) {
  const container = document.getElementById('recList');
  if (!top5 || top5.length === 0) {
    container.innerHTML = '<div class="empty-note">No recommendations available for this rank.</div>';
    return;
  }
  const weakWarning = sourceCollege && sourceCollege.Source_Match_Is_Weak
    ? `<div class="error-note">No college closely matches this rank (nearest cutoff is ${Math.round(sourceCollege.Source_Match_Gap)} rank positions away) — treat these recommendations with caution.</div>`
    : '';
  const shortListNote = top5Note ? `<div class="error-note">${top5Note}</div>` : '';
  container.innerHTML = weakWarning + shortListNote + top5.map(c => {
    const cutoff = pickCutoff(c);
    const tone = toneFor(c.Probability_Label);
    return `<div class="rec-row ${tone}">
      <div class="rec-main">
        <p class="college">${collegeShortName(c.College)}</p>
        <p class="loc">${collegeLoc(c.College)}</p>
      </div>
      <div class="cutoff">Cutoff ${cutoff != null ? Math.round(cutoff) : '—'}</div>
      <div class="pill ${tone}">${c.Probability_Label || 'Unknown'}</div>
    </div>`;
  }).join('');
}

function resetResults(ticketListHtml, recListHtml) {
  document.getElementById('bestCallout').innerHTML = '';
  document.getElementById('ticketList').innerHTML = ticketListHtml;
  document.getElementById('recList').innerHTML = recListHtml;
}

function setEvaluating(isEvaluating) {
  evaluating = isEvaluating;
  document.getElementById('evalBtn').disabled = isEvaluating;
}

async function evaluateNow() {
  if (evaluating) return;

  const rankInput = document.getElementById('rankInput');
  const rank = parseInt(rankInput.value, 10);

  if (!rank || rank <= 0) {
    resetResults('<div class="error-note">Enter a valid entrance rank (a positive number) before evaluating.</div>', '');
    return;
  }
  if (selectedNames.length === 0) {
    resetResults('<div class="empty-note">Add at least one college to your priority list before evaluating.</div>', '');
    return;
  }

  setEvaluating(true);
  resetResults('<div class="loading-note">Evaluating your chances…</div>', '<div class="loading-note">Loading recommendations…</div>');

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rank, preferences: selectedNames }),
    });

    if (!res.ok) {
      let detail = `Server returned ${res.status}`;
      try {
        const body = await res.json();
        if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      } catch (_) { /* response wasn't JSON */ }
      throw new Error(detail);
    }

    const data = await res.json();
    renderBestCallout(data.best_college);
    renderTicketList(data.preferences);
    renderRecommendations(data.top5_recommendations, data.source_college, data.top5_note);
  } catch (err) {
    document.getElementById('ticketList').innerHTML =
      `<div class="error-note">Could not evaluate your rank right now (${err.message}). Please try again.</div>`;
    document.getElementById('recList').innerHTML = '';
  } finally {
    setEvaluating(false);
  }
}

loadColleges();
