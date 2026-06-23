// fresh-test.cjs — tests ALL tools including piste/precis with Python bridge
const { spawn } = require('child_process');
const { resolve } = require('path');
const fs = require('fs');

const ROOT = resolve(__dirname);
const SERVER = resolve(ROOT, 'packages/server/src/index.ts');
const OUT = resolve(ROOT, 'tests', 'fresh-results.txt');

const log = [];
const add = (s) => { log.push(s); console.log(s); };

add('=== Fresh Test Suite ===\n');

const proc = spawn('node', ['--import', 'tsx', SERVER], {
  cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, MCP_LOG_LEVEL: 'warn', MCP_RATE_LIMIT_ENABLED: 'false' },
});

let stdout = '';
let stderr = '';
proc.stdout.on('data', (c) => { stdout += c.toString(); });
proc.stderr.on('data', (c) => { stderr += c.toString(); });

proc.on('error', (e) => add('SPAWN ERROR: ' + e.message));
proc.on('exit', (c) => add('Server exited: ' + c));

const send = (id, name, args = {}) => {
  proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id }) + '\n');
};

function getResponse(id, timeout = 30000) {
  return new Promise((resolve) => {
    const start = Date.now();
    const check = setInterval(() => {
      const lines = stdout.split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.id === id) { clearInterval(check); resolve(parsed); return; }
        } catch {}
      }
      if (Date.now() - start > timeout) { clearInterval(check); resolve(null); }
    }, 200);
  });
}

async function run() {
  await new Promise(r => setTimeout(r, 5000));
  add('Server started.\n');

  let id = 0;

  // ═══ Step 1: Verify server health ═══
  add('── Health Check ──');
  send(++id, 'mcp_health');
  const health = await getResponse(id);
  if (health && !health.error) {
    const d = JSON.parse(health.result.content[0].text);
    add('  PASS mcp_health: ' + d.tools.total + ' tools');
  } else {
    add('  FAIL mcp_health');
  }

  // ═══ Step 2: Test DeepPipe (always works) ═══
  add('\n── DeepPipe ──');
  send(++id, 'deeppipe_stats');
  send(++id, 'deeppipe_search', { query: 'test', limit: 3 });
  send(++id, 'deeppipe_ingest', { data: Buffer.from('Payment terms net 30 days. All invoices due within 15 days.').toString('base64'), source: 'invoice-test.txt' });
  send(++id, 'deeppipe_list_documents');

  for (const [name, rid] of [['stats', id-3], ['search', id-2], ['ingest', id-1], ['list', id]]) {
    const r = await getResponse(rid);
    const ok = r && !r.error && r.result && !r.result.isError;
    add('  ' + (ok ? 'PASS' : 'FAIL') + ' deeppipe_' + name);
  }

  // ═══ Step 3: Test Clinical ═══
  add('\n── Clinical Intake ──');
  send(++id, 'clinical_start_session', { lang: 'en' });
  send(++id, 'clinical_list_sessions');

  for (const [name, rid] of [['start_session', id-1], ['list_sessions', id]]) {
    const r = await getResponse(rid);
    const ok = r && !r.error && r.result && !r.result.isError;
    if (ok) {
      const d = JSON.parse(r.result.content[0].text);
      const detail = name === 'start_session' ? ('session: ' + (d.session_id || '?').slice(0, 12)) : (d.sessions?.length + ' sessions');
      add('  PASS clinical_' + name + ' (' + detail + ')');
    } else {
      const err = r?.error?.message || r?.result?.content?.[0]?.text || 'no response';
      add('  FAIL clinical_' + name + ': ' + err.slice(0, 60));
    }
  }

  // ═══ Step 4: Test Piste (Python bridge) ═══
  add('\n── Piste (Python DSPy pipeline) ──');
  send(++id, 'piste_fact_check', { claim_text: 'The sky is blue because of Rayleigh scattering of sunlight.', locale: 'en' });

  const pisteR = await getResponse(id, 60000);
  if (pisteR && !pisteR.error && pisteR.result && !pisteR.result.isError) {
    const d = JSON.parse(pisteR.result.content[0].text);
    const v = d.verdict?.label || d.note || '?';
    add('  PASS piste_fact_check: verdict=' + v);
  } else {
    const err = pisteR?.error?.message || pisteR?.result?.content?.[0]?.text || 'no response (timeout)';
    add('  FAIL piste_fact_check: ' + err.slice(0, 80));
  }

  // ═══ Step 5: Test Precis (Python bridge) ═══
  add('\n── Precis (Python RAG pipeline) ──');
  send(++id, 'precis_list_documents');
  send(++id, 'precis_debug_stem', { q: 'payment terms invoice' });

  for (const [name, rid] of [['list_documents', id-1], ['debug_stem', id]]) {
    const r = await getResponse(rid);
    const ok = r && !r.error && r.result && !r.result.isError;
    if (ok) {
      const d = JSON.parse(r.result.content[0].text);
      add('  PASS precis_' + name + ' (' + (Array.isArray(d) ? d.length + ' items' : d.stemmed_tokens?.length + ' tokens') + ')');
    } else {
      const err = r?.error?.message || r?.result?.content?.[0]?.text || 'no response';
      add('  FAIL precis_' + name + ': ' + err.slice(0, 60));
    }
  }

  // ═══ Show Python status ═══
  add('\n── Python Bridge Status ──');
  const pyLogs = stderr.split('\n').filter(l => l.includes('Python') || l.includes('python'));
  if (pyLogs.length > 0) {
    for (const l of pyLogs.slice(0, 10)) {
      try { const p = JSON.parse(l); add('  ' + p.level + ': ' + p.message); } catch { add('  ' + l); }
    }
  } else {
    add('  (no Python-related logs found)');
  }

  proc.kill();

  // ═══ Summary ═══
  const results = log.join('\n');
  const passCount = (results.match(/PASS/g) || []).length;
  const failCount = (results.match(/FAIL/g) || []).length;
  add('\n=== ' + passCount + ' PASS, ' + failCount + ' FAIL ===');
  fs.writeFileSync(OUT, results);
  add('Saved to tests/fresh-results.txt');
}

run().catch(e => { add('FATAL: ' + e.message); proc.kill(); });
