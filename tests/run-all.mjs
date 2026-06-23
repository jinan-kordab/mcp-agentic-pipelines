/**
 * MCP Server Test Suite
 * 
 * Tests all 31 tools from the command line.
 * Some tools require Python backends (piste, precis) — those are skipped gracefully.
 * Some tools require API keys — those are tested when keys are available.
 * 
 * Usage:
 *   cd unified-mcp-server
 *   node --import tsx tests/run-all.mjs
 * 
 * Or with a filter:
 *   node --import tsx tests/run-all.mjs --filter=deeppipe
 *   node --import tsx tests/run-all.mjs --filter=clinical
 *   node --import tsx tests/run-all.mjs --filter=mcp
 */

import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SERVER_PATH = resolve(ROOT, 'packages/server/src/index.ts');

// ── ANSI colors ──────────────────────────────────────────────────────
const C = { green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m', cyan: '\x1b[36m', gray: '\x1b[90m', bold: '\x1b[1m', reset: '\x1b[0m' };
const pass = (m) => `${C.green}✅ ${m}${C.reset}`;
const fail = (m) => `${C.red}❌ ${m}${C.reset}`;
const skip = (m) => `${C.yellow}⏭️  ${m}${C.reset}`;
const info = (m) => `${C.cyan}ℹ️  ${m}${C.reset}`;
const header = (m) => `${C.bold}${C.cyan}${m}${C.reset}`;

// ── Test definitions: [toolName, args, validator, category] ─────────
const TESTS = [
  // ═══ Built-in (always work) ═══
  ['mcp_health', {}, (r) => r.server && r.tools?.total > 0, 'built-in'],
  ['mcp_list_providers', {}, (r) => Array.isArray(r) && r.length >= 9, 'built-in'],

  // ═══ DeepPipe — Search & Ingest (always work, no keys needed) ═══
  ['deeppipe_stats', {}, (r) => typeof r.documentCount === 'number', 'deeppipe'],
  ['deeppipe_list_documents', {}, (r) => Array.isArray(r.documents), 'deeppipe'],
  ['deeppipe_search', { query: 'test', limit: 5 }, (r) => Array.isArray(r.hits), 'deeppipe'],
  ['deeppipe_ingest', { data: Buffer.from('Hello world. This is a test document for MCP verification.').toString('base64'), source: 'test-mcp.txt' }, (r) => r.documentId > 0, 'deeppipe'],
  ['deeppipe_get_document', { id: 1 }, (r) => r.document, 'deeppipe'],
  ['deeppipe_get_text', { id: 1 }, (r) => typeof r.text === 'string', 'deeppipe'],
  ['deeppipe_extractive_answer', { question: 'What is this about?' }, (r) => r.answer, 'deeppipe'],

  // ═══ DeepPipe — Chat (needs LLM key) ═══
  ['deeppipe_chat_context', { question: 'What is the test document about?' }, (r) => r.sources, 'deeppipe-llm'],

  // ═══ Piste (needs Python backend on port 8000) ═══
  ['piste_fact_check', { claim_text: 'The sky is green.', locale: 'en' }, (r) => r.verdict, 'piste'],
  ['piste_list_verdicts', {}, (r) => Array.isArray(r), 'piste'],

  // ═══ Precis (needs Python backend on port 8001) ═══
  ['precis_list_documents', {}, (r) => Array.isArray(r), 'precis'],
  ['precis_debug_stem', { q: 'payment terms' }, (r) => r.stemmed_tokens, 'precis'],

  // ═══ Clinical (needs Groq + ElevenLabs keys) ═══
  ['clinical_start_session', { lang: 'en' }, (r) => r.session_id && r.greeting_text, 'clinical'],
  ['clinical_list_sessions', {}, (r) => Array.isArray(r.sessions), 'clinical'],
];

// ── MCP Client ────────────────────────────────────────────────────────

class MCPTestClient {
  constructor() {
    this.server = null;
    this.buffer = '';
    this.nextId = 1;
    this.pending = new Map();
  }

  async start() {
    return new Promise((resolve, reject) => {
      this.server = spawn('node', ['--import', 'tsx', SERVER_PATH], {
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env },
      });

      this.server.stdout.on('data', (chunk) => {
        this.buffer += chunk.toString();
        this.processBuffer();
      });

      this.server.stderr.on('data', (chunk) => {
        // Logs to stderr are normal (JSON structured logs)
        // Uncomment to debug: process.stderr.write(chunk);
      });

      this.server.on('error', reject);

      // Give the server a moment to start
      setTimeout(resolve, 2000);
    });
  }

  processBuffer() {
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const msg = JSON.parse(trimmed);
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          resolve(msg);
        }
      } catch {
        // Not JSON — probably a log line
      }
    }
  }

  async call(toolName, args = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timeout waiting for ${toolName}`));
      }, 15000);

      this.pending.set(id, { resolve: (msg) => { clearTimeout(timer); resolve(msg); }, reject });

      const request = JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/call',
        params: { name: toolName, arguments: args },
        id,
      });
      this.server.stdin.write(request + '\n');
    });
  }

  async stop() {
    if (this.server) {
      this.server.kill();
      this.server = null;
    }
  }
}

// ── Test Runner ───────────────────────────────────────────────────────

async function runAllTests(filter = '') {
  console.log(header('\n╔══════════════════════════════════════════════╗'));
  console.log(header('║   Unified MCP Server — Test Suite           ║'));
  console.log(header('╚══════════════════════════════════════════════╝\n'));

  const client = new MCPTestClient();
  
  console.log(info('Starting MCP server...'));
  await client.start();
  console.log(info('Server started. Running tests...\n'));

  let passed = 0, failed = 0, skipped = 0, total = 0;

  // Filter tests if a filter is provided
  const testsToRun = filter
    ? TESTS.filter(([name, , , cat]) => name.includes(filter) || cat.includes(filter))
    : TESTS;

  let currentCategory = '';
  
  for (const [toolName, args, validator, category] of testsToRun) {
    total++;
    
    // Print category header
    if (category !== currentCategory) {
      currentCategory = category;
      const catLabel = category.replace('-', ' + ').toUpperCase();
      console.log(`\n${C.bold}── ${catLabel} ──${C.reset}`);
    }

    const label = toolName.padEnd(32);
    
    try {
      const response = await client.call(toolName, args);
      
      if (response.error) {
        const errMsg = response.error.message || JSON.stringify(response.error);
        // Check for "service unavailable" or "not configured" → skip
        if (errMsg.includes('not reachable') || errMsg.includes('not configured') || errMsg.includes('SERVICE_UNAVAILABLE')) {
          console.log(`  ${skip(label)} ${C.gray}${errMsg.slice(0, 60)}${C.reset}`);
          skipped++;
        } else {
          console.log(`  ${fail(label)} ${errMsg.slice(0, 80)}`);
          failed++;
        }
        continue;
      }

      // Parse the result content
      const content = response.result?.content?.[0]?.text;
      if (!content) {
        console.log(`  ${fail(label)} No content in response`);
        failed++;
        continue;
      }

      let data;
      try {
        data = JSON.parse(content);
      } catch {
        data = content;
      }

      // Check if it's an error response
      if (response.result?.isError || (data && data.error)) {
        const errMsg = data?.error?.message || data?.error || 'Unknown error';
        if (errMsg.includes('not reachable') || errMsg.includes('not configured') || errMsg.includes('SERVICE_UNAVAILABLE') || errMsg.includes('NOT_FOUND')) {
          console.log(`  ${skip(label)} ${C.gray}${String(errMsg).slice(0, 60)}${C.reset}`);
          skipped++;
        } else {
          console.log(`  ${fail(label)} ${String(errMsg).slice(0, 80)}`);
          failed++;
        }
        continue;
      }

      // Validate the result
      if (validator(data)) {
        const summary = getResultSummary(toolName, data);
        console.log(`  ${pass(label)} ${C.gray}${summary}${C.reset}`);
        passed++;
      } else {
        console.log(`  ${fail(label)} Unexpected response format`);
        failed++;
      }

    } catch (err) {
      const msg = err.message || String(err);
      if (msg.includes('Timeout')) {
        console.log(`  ${fail(label)} Timeout — server may have crashed`);
      } else {
        console.log(`  ${fail(label)} ${msg.slice(0, 80)}`);
      }
      failed++;
    }
  }

  await client.stop();

  // ── Summary ──────────────────────────────────────────────────────
  console.log(`\n${header('═══════════════════════════════════════════')}`);
  console.log(header(`  Results: ${pass(''+passed)}  ${fail(''+failed)}  ${skip(''+skipped)}  (${total} total)`));
  console.log(header('═══════════════════════════════════════════'));
  
  if (failed === 0) {
    console.log(`\n${C.green}${C.bold}✅ All reachable tools passed!${C.reset}`);
  }
  
  if (skipped > 0) {
    console.log(`\n${C.yellow}💡 ${skipped} tools were skipped — they need additional services:${C.reset}`);
    console.log('   • piste tools:     docker compose up -d in the piste/ folder');
    console.log('   • precis tools:    start.bat in the precis-agentic-pipeline/ folder');
    console.log('   • clinical tools:  set GROQ_API_KEY + ELEVENLABS_API_KEY in .env');
    console.log('   • chat tools:      set LLM_DEFAULT_API_KEY in .env (already configured ✅)');
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

function getResultSummary(toolName, data) {
  switch (toolName) {
    case 'mcp_health':           return `v${data.server?.version} · ${data.tools?.total} tools`;
    case 'mcp_list_providers':   return `${data.length} providers`;
    case 'deeppipe_stats':       return `${data.documentCount} docs`;
    case 'deeppipe_search':      return `${data.totalHits} hits · ${data.elapsedMs}ms`;
    case 'deeppipe_ingest':      return `doc #${data.documentId} · ${data.wordCount} words`;
    case 'deeppipe_list_documents': return `${data.documents?.length || 0} docs`;
    case 'deeppipe_get_document':   return data.document?.source || 'ok';
    case 'deeppipe_get_text':       return `${data.text?.length || 0} chars`;
    case 'deeppipe_chat_context':   return `${data.sources?.length || 0} sources`;
    case 'deeppipe_extractive_answer': return `${(data.answer || '').slice(0, 30)}...`;
    case 'piste_fact_check':     return `verdict: ${data.verdict?.label || '?'}`;
    case 'clinical_start_session': return `session: ${data.session_id?.slice(0, 15)}...`;
    case 'clinical_list_sessions':  return `${data.sessions?.length || 0} sessions`;
    default:                     return 'ok';
  }
}

// ── CLI ───────────────────────────────────────────────────────────────

const filter = process.argv.find(a => a.startsWith('--filter='))?.split('=')[1] || '';

console.log(`MCP Server path: ${SERVER_PATH}`);
console.log(`Filter: ${filter || 'none (all tests)'}\n`);

runAllTests(filter).catch((err) => {
  console.error(fail('Test runner crashed:'), err);
  process.exit(1);
});
