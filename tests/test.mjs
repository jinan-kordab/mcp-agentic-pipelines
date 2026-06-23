/**
 * One-shot test — spawns server, sends requests, saves output to tests/results.txt
 * Run: node --import tsx tests/test.mjs
 */
import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SERVER_PATH = resolve(ROOT, 'packages/server/src/index.ts');
const RESULTS_FILE = resolve(__dirname, 'results.txt');

async function run() {
  const server = spawn('node', ['--import', 'tsx', SERVER_PATH], {
    cwd: ROOT,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env },
  });

  const log = [];
  const add = (line) => { log.push(line); process.stdout.write(line + '\n'); };

  let stdout = '';
  server.stdout.on('data', (c) => { stdout += c.toString(); });
  server.stderr.on('data', (c) => { /* ignore dotenv noise */ });

  await new Promise(r => setTimeout(r, 3000));
  add('Server started, sending requests...\n');

  const tests = [
    ['mcp_health', {}],
    ['mcp_list_providers', {}],
    ['deeppipe_stats', {}],
    ['deeppipe_list_documents', {}],
    ['deeppipe_search', { query: 'test', limit: 3 }],
  ];

  for (let i = 0; i < tests.length; i++) {
    const [name, args] = tests[i];
    const req = JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id: i + 1 });
    server.stdin.write(req + '\n');
    await new Promise(r => setTimeout(r, 1500));
  }

  await new Promise(r => setTimeout(r, 3000));
  server.kill();

  // Parse responses
  const responses = stdout.split('\n').filter(l => l.trim()).map(l => {
    try { return JSON.parse(l); } catch { return null; }
  }).filter(Boolean);

  add(`\nResponses parsed: ${responses.length}\n`);

  for (const r of responses) {
    const name = tests[r.id - 1]?.[0] || `id=${r.id}`;
    const ok = !r.error && r.result && !r.result.isError;
    const summary = ok ? 'PASS' : `FAIL: ${r.error?.message || r.result?.content?.[0]?.text || 'unknown'}`;
    add(`  ${ok ? '✅' : '❌'} ${name}: ${summary}`);
  }

  mkdirSync(dirname(RESULTS_FILE), { recursive: true });
  writeFileSync(RESULTS_FILE, log.join('\n'));
  add(`\nResults saved to: ${RESULTS_FILE}`);
}

run().catch(e => console.error('CRASH:', e));
