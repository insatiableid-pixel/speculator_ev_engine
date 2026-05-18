/* speculator_ev_engine — vanilla JS, no framework, fetch + DOM patch */

const API = '';

// Tab switching
document.querySelectorAll('#nav button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// Helpers
async function post(endpoint, body) {
  const r = await fetch(API + endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

function fmtEV(v) {
  const s = v >= 0 ? '+' : '';
  return s + v.toFixed(3);
}

function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }
function fmtDollar(v) { return (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2); }

function colorEV(v) {
  if (v > 0.001) return 'ev-pos';
  if (v < -0.001) return 'ev-neg';
  return '';
}

function updateContext(data) {
  if (data.fraction !== undefined)
    document.getElementById('ctx-kelly').textContent = 'Kelly: ' + fmtPct(data.fraction);
  if (data.expected_log_growth !== undefined)
    document.getElementById('ctx-ev').textContent = 'Session EV: ' + fmtEV(data.expected_log_growth);
}

// Kelly
document.getElementById('kelly-go').addEventListener('click', async () => {
  const p = parseFloat(document.getElementById('k-p').value);
  const b = parseFloat(document.getElementById('k-b').value);
  const frac = parseFloat(document.getElementById('k-f').value);
  const es = parseFloat(document.getElementById('k-es').value);

  let out = '';
  try {
    const r = await post('/kelly/binary', { p, b });
    out += `Full: f=${fmtPct(r.fraction)}  g=${fmtEV(r.expected_log_growth)}  ruin=${fmtPct(r.ruin_probability)}\n`;

    if (frac < 1) {
      const r2 = await post('/kelly/fractional', { p, b, fraction: frac });
      out += `${frac*100}% Kelly: f=${fmtPct(r2.fraction)}  g=${fmtEV(r2.expected_log_growth)}  ruin=${fmtPct(r2.ruin_probability)}\n`;
    }

    if (es > 0) {
      const edge = p - (1 - p) / b;
      const r3 = await post('/kelly/uncertain-edge', { edge_mean: edge, edge_std: es, odds: b });
      out += `Uncertain: f=${fmtPct(r3.fraction)}  g=${fmtEV(r3.expected_log_growth)}  ruin=${fmtPct(r3.ruin_probability)}\n`;
    }

    updateContext(r);
  } catch (e) {
    out += 'Error: ' + e.message;
  }

  document.getElementById('kelly-out').textContent = out;
});

// ICM
document.getElementById('icm-go').addEventListener('click', async () => {
  const stacks = document.getElementById('i-stacks').value.split(',').map(Number);
  const payouts = document.getElementById('i-payouts').value.split(',').map(Number);
  const blend = parseFloat(document.getElementById('i-blend').value);

  try {
    const r = await post('/icm/equity', { stacks, payouts, blend });
    let out = `Prize pool: $${r.total_prize_pool.toFixed(0)}\n`;
    r.equities.forEach((eq, i) => {
      out += `  Stack ${i+1} (${stacks[i].toLocaleString()}): $${eq.toFixed(2)}\n`;
    });
    document.getElementById('icm-out').textContent = out;
  } catch (e) {
    document.getElementById('icm-out').textContent = 'Error: ' + e.message;
  }
});

// Sports Edge
document.getElementById('sports-edge-go').addEventListener('click', async () => {
  const mp = parseFloat(document.getElementById('s-mp').value);
  const odds = parseInt(document.getElementById('s-odds').value);
  const other = parseInt(document.getElementById('s-other').value);

  try {
    const r = await post('/sports/edge', { model_prob: mp, odds_american: odds, other_outcomes: [other] });
    const out = `Edge: ${fmtPct(r.edge)}  Model: ${fmtPct(r.model_prob)}  Market: ${fmtPct(r.market_prob)}  EV/unit: ${fmtEV(r.ev_per_unit)}`;
    document.getElementById('sports-out').textContent = out;
    updateContext(r);
  } catch (e) {
    document.getElementById('sports-out').textContent = 'Error: ' + e.message;
  }
});

// Sports CLV
document.getElementById('sports-clv-go').addEventListener('click', async () => {
  const open = parseInt(document.getElementById('s-open').value);
  const close = parseInt(document.getElementById('s-close').value);

  try {
    const r = await post('/sports/clv', { open_odds: open, close_odds: close });
    const out = `CLV: ${fmtPct(r.clv)}  Open imp: ${fmtPct(r.open_implied)}  Close imp: ${fmtPct(r.close_implied)}`;
    document.getElementById('sports-out').textContent += '\n' + out;
  } catch (e) {
    document.getElementById('sports-out').textContent = 'Error: ' + e.message;
  }
});

// Decisions
document.getElementById('dec-log-go').addEventListener('click', async () => {
  const dec = document.getElementById('d-dec').value;
  const p = parseFloat(document.getElementById('d-p').value);
  const ev = parseFloat(document.getElementById('d-ev').value);
  const stake = parseFloat(document.getElementById('d-stake').value);
  const domain = document.getElementById('d-domain').value;

  try {
    const r = await post('/decisions/log', { decision: dec, p_estimate: p, ev_estimate: ev, stake, domain });
    document.getElementById('decisions-out').textContent = `Logged as row ${r.row_id}`;
  } catch (e) {
    document.getElementById('decisions-out').textContent = 'Error: ' + e.message;
  }
});

document.getElementById('dec-resolve-go').addEventListener('click', async () => {
  const rid = parseInt(document.getElementById('d-rid').value);
  const out = parseFloat(document.getElementById('d-outcome').value);

  try {
    const r = await post('/decisions/resolve', { row_id: rid, outcome: out });
    document.getElementById('decisions-out').textContent = r.status;
  } catch (e) {
    document.getElementById('decisions-out').textContent = 'Error: ' + e.message;
  }
});
