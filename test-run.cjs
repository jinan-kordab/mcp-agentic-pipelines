// test-runner.cjs — CommonJS, writes to file, no tsx needed
const { spawn } = require('child_process');
const { resolve, dirname } = require('path');
const fs = require('fs');

const ROOT = resolve(__dirname);
const SERVER = resolve(ROOT, 'packages/server/src/index.ts');
const OUT = resolve(ROOT, 'tests', 'test-results.txt');
fs.mkdirSync(resolve(ROOT, 'tests'), { recursive: true });

const log = [];
const add = (s) => { log.push(s); process.stdout.write(s + '\n'); };

add('Starting MCP server...');
const proc = spawn('node', ['--import', 'tsx', SERVER], {
  cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, MCP_LOG_LEVEL: 'error' },
});

let stdout = '';
proc.stdout.on('data', (c) => { stdout += c.toString(); });
proc.stderr.on('data', () => {}); // ignore

setTimeout(async () => {
  add('Sending requests...\n');

  const tests = [
    ['mcp_health', {}],
    ['mcp_list_providers', {}],
    ['deeppipe_stats', {}],
    ['deeppipe_list_documents', {}],
    ['deeppipe_search', { query: 'test', limit: 3 }],
    ['deeppipe_ingest', { data: Buffer.from('Hello world test').toString('base64'), source: 't.txt' }],
    ['deeppipe_get_document', { id: 1 }],
    ['deeppipe_get_text', { id: 1 }],
    ['deeppipe_extractive_answer', { question: 'test' }],
    ['piste_list_verdicts', {}],
    ['precis_list_documents', {}],
    ['clinical_list_sessions', {}],
  ];

  for (let i = 0; i < tests.length; i++) {
    const [name, args] = tests[i];
    proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id: i + 1 }) + '\n');
    await new Promise(r => setTimeout(r, 1500));
  }

  await new Promise(r => setTimeout(r, 3000));
  proc.kill();

  const responses = stdout.split('\n').filter(l => l.trim()).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  
  let p = 0, f = 0;
  for (let i = 0; i < tests.length; i++) {
    const r = responses.find(x => x.id === i + 1);
    const ok = r && !r.error && r.result && !r.result.isError;
    if (ok) { add('  PASS ' + tests[i][0]); p++; }
    else { add('  FAIL ' + tests[i][0] + ' ' + (r?.error?.message || r?.result?.content?.[0]?.text || 'no response').slice(0,60)); f++; }
  }
  
  add('\n=== ' + p + '/' + tests.length + ' passed, ' + f + ' failed ===');
  fs.writeFileSync(OUT, log.join('\n'));
  process.exit(0);
}, 4000);
