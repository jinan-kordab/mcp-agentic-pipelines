/**
 * Quick Smoke Test — tests only the tools that ALWAYS work
 * (no API keys, no Python backends, no external services).
 * 
 * Usage: node --import tsx tests/smoke.mjs
 */

import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SERVER_PATH = resolve(ROOT, 'packages/server/src/index.ts');

const PASS = '\x1b[32m✅\x1b[0m';
const FAIL = '\x1b[31m❌\x1b[0m';

async function smokeTest() {
  const server = spawn('node', ['--import', 'tsx', SERVER_PATH], {
    cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env },
  });

  let stdout = '';
  server.stdout.on('data', (c) => { stdout += c.toString(); });

  await new Promise(r => setTimeout(r, 3000));

  const tools = [
    ['mcp_health', {}],
    ['mcp_list_providers', {}],
    ['deeppipe_stats', {}],
    ['deeppipe_list_documents', {}],
    ['deeppipe_search', { query: 'test', limit: 3 }],
  ];

  let id = 1;
  for (const [name, args] of tools) {
    server.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id: id++ }) + '\n');
  }

  await new Promise(r => setTimeout(r, 5000));
  server.kill();

  const responses = stdout.split('\n').filter(l => l.trim()).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  
  console.log(`\nResponses: ${responses.length}/${tools.length}\n`);
  
  for (const r of responses) {
    const name = tools[r.id - 1]?.[0] || `id=${r.id}`;
    const ok = !r.error && r.result && !r.result.isError;
    console.log(`  ${ok ? PASS : FAIL} ${name}: ${ok ? 'OK' : (r.error?.message || 'FAILED')}`);
  }
}

smokeTest().catch(console.error);
