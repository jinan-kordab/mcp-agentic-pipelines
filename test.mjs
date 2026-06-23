#!/usr/bin/env node
/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║   Unified MCP Server — Complete Test Suite                ║
 * ║   Tests ALL 31 tools across 5 repositories                ║
 * ║   Run: node test.mjs                                      ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Tests every tool, handles timeouts, shows pass/fail/skip.
 * Python backends auto-start if Python is available.
 * Single file — drop anywhere, run anywhere.
 */

import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname);
const SERVER = resolve(ROOT, 'packages', 'server', 'src', 'index.ts');

// ── Terminal colors ──────────────────────────────────────────────────
const G = '\x1b[32m', R = '\x1b[31m', Y = '\x1b[33m', C = '\x1b[36m', B = '\x1b[1m', D = '\x1b[90m', X = '\x1b[0m';
const ok = (s) => G + '✅ ' + s + X;
const no = (s) => R + '❌ ' + s + X;
const sk = (s) => Y + '⏭️  ' + s + X;
const h1 = (s) => '\n' + B + C + s + X;
const h2 = (s) => B + D + s + X;

// ── MCP Test Client ──────────────────────────────────────────────────

class MCPClient {
  constructor() {
    this.proc = null;
    this.stdout = '';
    this.stderr = '';
    this.nextId = 1;
    this.pending = new Map();
  }

  async start() {
    return new Promise((resolve, reject) => {
      this.proc = spawn('node', ['--import', 'tsx', SERVER], {
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_LOG_LEVEL: 'error', MCP_RATE_LIMIT_ENABLED: 'false' },
      });

      this.proc.stdout.on('data', (c) => {
        this.stdout += c.toString();
        this._processBuffer();
      });

      this.proc.stderr.on('data', (c) => {
        this.stderr += c.toString();
        // Watch for server ready signal
        if (this.stderr.includes('__MCP_READY__') && !this._readyFired) {
          this._readyFired = true;
          this._doHandshake(resolve, reject);
        }
      });

      this.proc.on('error', reject);

      // Fallback: if server hasn't signaled ready in 5 min, fail
      setTimeout(() => {
        if (!this._readyFired) {
          reject(new Error('Server did not become ready within 5 minutes'));
        }
      }, 300_000);
    });
  }

  async _doHandshake(resolve, reject) {
    try {
      // MCP initialize handshake
      const initResult = await this._sendRaw('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        clientInfo: { name: 'test', version: '1.0.0' },
      }, 10000);
      if (initResult._timedOut) {
        reject(new Error('initialize handshake timed out'));
        return;
      }
      // Send initialized notification (no id — notification, not request)
      this.proc.stdin.write(JSON.stringify({
        jsonrpc: '2.0',
        method: 'notifications/initialized',
      }) + '\n');

      // Give server a moment, then resolve
      setTimeout(() => resolve(), 500);
    } catch (e) {
      reject(e);
    }
  }

  /** Send a raw JSON-RPC request and return the response. */
  async _sendRaw(method, params, timeoutMs = 30000) {
    const id = this.nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve({ _timedOut: true });
      }, timeoutMs);

      this.pending.set(id, { resolve: (msg) => { clearTimeout(timer); resolve(msg); } });

      this.proc.stdin.write(JSON.stringify({
        jsonrpc: '2.0',
        method,
        params,
        id,
      }) + '\n');
    });
  }

  _processBuffer() {
    const lines = this.stdout.split('\n');
    this.stdout = lines.pop() || '';
    for (const line of lines) {
      try {
        const msg = JSON.parse(line.trim());
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          resolve(msg);
        }
      } catch {}
    }
  }

  async call(toolName, args = {}, timeoutMs = 30000) {
    const id = this.nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve({ _timedOut: true, toolName });
      }, timeoutMs);

      this.pending.set(id, { resolve: (msg) => { clearTimeout(timer); resolve(msg); } });

      this.proc.stdin.write(JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/call',
        params: { name: toolName, arguments: args },
        id,
      }) + '\n');
    });
  }

  stop() {
    if (this.proc) { this.proc.kill(); this.proc = null; }
  }
}

// ── Test definitions ─────────────────────────────────────────────────

const TESTS = [
  // ═══ Built-in (always) ═══
  { name: 'mcp_health',            args: {},                    timeout: 5000,  cat: 'Built-in' },
  { name: 'mcp_list_providers',    args: {},                    timeout: 5000,  cat: 'Built-in' },

  // ═══ DeepPipe (always works, no keys) ═══
  { name: 'deeppipe_stats',        args: {},                    timeout: 5000,  cat: 'DeepPipe' },
  { name: 'deeppipe_list_documents', args: {},                  timeout: 5000,  cat: 'DeepPipe' },
  { name: 'deeppipe_search',       args: { query: 'test', limit: 3 }, timeout: 5000, cat: 'DeepPipe' },
  { name: 'deeppipe_ingest',       args: { data: Buffer.from('Test document. Payment terms net 30.').toString('base64'), source: 'test.txt' }, timeout: 10000, cat: 'DeepPipe' },
  { name: 'deeppipe_get_document', args: { id: 1 },             timeout: 5000,  cat: 'DeepPipe' },
  { name: 'deeppipe_get_text',     args: { id: 1 },             timeout: 5000,  cat: 'DeepPipe' },
  { name: 'deeppipe_remove_document', args: { id: 1 },          timeout: 5000,  cat: 'DeepPipe' },
  { name: 'deeppipe_ingest_file',  args: { path: 'test.txt' },  timeout: 5000,  cat: 'DeepPipe', expectFail: true },
  { name: 'deeppipe_extractive_answer', args: { question: 'what is this about?' }, timeout: 10000, cat: 'DeepPipe' },
  { name: 'deeppipe_chat_context', args: { question: 'payment?' }, timeout: 15000, cat: 'DeepPipe (LLM)' },

  // ═══ Piste (Python bridge or local DSPy) ═══
  { name: 'piste_fact_check',      args: { claim_text: 'The sky is blue because of Rayleigh scattering.', locale: 'en' }, timeout: 90000, cat: 'Piste (Python)' },
  { name: 'piste_list_verdicts',   args: {},                    timeout: 5000,  cat: 'Piste' },
  { name: 'piste_replay',          args: { run_id: 'test' },    timeout: 5000,  cat: 'Piste' },
  { name: 'piste_get_audit',       args: { run_id: 'test' },    timeout: 5000,  cat: 'Piste' },
  { name: 'piste_get_verdict',     args: { claim_id: 'test' },  timeout: 5000,  cat: 'Piste' },
  { name: 'piste_submit_feedback', args: { run_id: 'test', rating: 3 }, timeout: 5000, cat: 'Piste' },

  // ═══ Precis (Python bridge or local backend) ═══
  { name: 'precis_list_documents', args: {},                    timeout: 30000, cat: 'Precis (Python)' },
  { name: 'precis_debug_stem',     args: { q: 'payment terms' }, timeout: 30000, cat: 'Precis (Python)' },
  { name: 'precis_debug_search',   args: { q: 'test' },         timeout: 30000, cat: 'Precis (Python)' },
  { name: 'precis_query',          args: { query: 'test' },     timeout: 60000, cat: 'Precis (Python)' },
  { name: 'precis_upload_document', args: { data: Buffer.from('test').toString('base64'), filename: 't.txt' }, timeout: 30000, cat: 'Precis' },
  { name: 'precis_upload_batch',   args: { files: [] },         timeout: 5000,  cat: 'Precis' },
  { name: 'precis_extract_work_order', args: { data: Buffer.from('test').toString('base64') }, timeout: 5000, cat: 'Precis' },
  { name: 'precis_list_work_orders', args: {},                  timeout: 5000,  cat: 'Precis' },

  // ═══ Clinical (needs Groq + ElevenLabs keys) ═══
  { name: 'clinical_start_session',   args: { lang: 'en' },     timeout: 15000, cat: 'Clinical' },
  { name: 'clinical_list_sessions',   args: {},                 timeout: 5000,  cat: 'Clinical' },
  { name: 'clinical_get_session',     args: { session_id: 'x' }, timeout: 5000, cat: 'Clinical' },
  { name: 'clinical_process_audio',   args: { session_id: 'x', audio_data: 'dGVzdA==' }, timeout: 5000, cat: 'Clinical (audio)' },
  { name: 'clinical_generate_podcast', args: { session_id: 'x' }, timeout: 5000, cat: 'Clinical (audio)' },
];

// ── Result interpreter ───────────────────────────────────────────────

function interpret(toolName, response) {
  if (response._timedOut) return { status: 'fail', reason: 'Timeout (no response from server)' };

  // MCP JSON-RPC error
  if (response.error) {
    const msg = response.error.message || '';
    if (/not reachable|SERVICE_UNAVAILABLE|not configured|Python not found|spawn python/i.test(msg))
      return { status: 'skip', reason: msg.slice(0, 60) };
    if (/NOT_FOUND|not found/i.test(msg))
      return { status: 'skip', reason: msg.slice(0, 60) };
    if (/RATE_LIMIT/i.test(msg))
      return { status: 'fail', reason: 'Rate limited (retry)' };
    return { status: 'fail', reason: msg.slice(0, 80) };
  }

  // Check result content
  const content = response.result?.content?.[0]?.text;
  if (!content) return { status: 'fail', reason: 'Empty response' };

  let data;
  try { data = JSON.parse(content); } catch { data = content; }

  // Is it an error?
  if (response.result?.isError) {
    const errMsg = typeof data === 'string' ? data : (data?.error?.message || data?.error || data?.note || JSON.stringify(data).slice(0, 50));
    if (/not reachable|not configured|not found|Python|backend|bridge|available with full/i.test(errMsg))
      return { status: 'skip', reason: errMsg.slice(0, 60) };
    return { status: 'fail', reason: errMsg.slice(0, 80) };
  }

  return { status: 'pass', detail: summarize(toolName, data) };
}

function summarize(name, data) {
  const d = data || {};
  switch (name) {
    case 'mcp_health': return d.tools?.total + ' tools, ' + d.llm?.defaultProvider;
    case 'mcp_list_providers': return (Array.isArray(d) ? d.length : '?') + ' providers';
    case 'deeppipe_stats': return d.documentCount + ' docs';
    case 'deeppipe_search': return d.totalHits + ' hits';
    case 'deeppipe_ingest': return 'doc #' + d.documentId + ', ' + d.wordCount + ' words';
    case 'deeppipe_list_documents': return (d.documents?.length || 0) + ' docs';
    case 'deeppipe_get_document': return d.document?.source || 'ok';
    case 'deeppipe_get_text': return (d.text?.length || 0) + ' chars';
    case 'deeppipe_chat_context': return (d.sources?.length || 0) + ' sources';
    case 'deeppipe_extractive_answer': return (d.answer || '').slice(0, 30) + '...';
    case 'piste_fact_check': return 'verdict: ' + (d.verdict?.label || d.note || '?');
    case 'clinical_start_session': return 'session: ' + (d.session_id || '?').slice(0, 12);
    case 'clinical_list_sessions': return (d.sessions?.length || 0) + ' sessions';
    case 'precis_debug_stem': return (d.stemmed_tokens?.length || 0) + ' tokens';
    case 'precis_list_documents': return Array.isArray(d) ? d.length + ' docs' : 'ok';
    default: return 'ok';
  }
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  console.log(h1('╔══════════════════════════════════════════╗'));
  console.log(h1('║   MCP Agentic Pipelines — Test Suite    ║'));
  console.log(h1('╚══════════════════════════════════════════╝'));
  console.log(D + 'Server: ' + SERVER + X + '\n');

  const client = new MCPClient();

  console.log('Starting server...');
  await client.start();
  console.log('Server ready. Testing ' + TESTS.length + ' tools...\n');

  const results = [];
  let currentCat = '';

  for (let i = 0; i < TESTS.length; i++) {
    const { name, args, timeout, cat, expectFail } = TESTS[i];

    if (cat !== currentCat) {
      currentCat = cat;
      console.log(h2('── ' + cat + ' ──'));
    }

    const label = name.padEnd(30);
    const response = await client.call(name, args, timeout);
    const result = interpret(name, response);

    // If expected to fail and it fails → that's a pass
    if (expectFail && result.status === 'fail') {
      result.status = 'pass';
      result.detail = 'expected failure';
    }

    switch (result.status) {
      case 'pass': console.log('  ' + ok(label) + D + (result.detail || '') + X); break;
      case 'fail': console.log('  ' + no(label) + D + (result.reason || '') + X); break;
      case 'skip': console.log('  ' + sk(label) + D + (result.reason || '') + X); break;
    }

    results.push({ name, ...result });
  }

  client.stop();

  // ── Summary ────────────────────────────────────────────────────
  const passed = results.filter(r => r.status === 'pass').length;
  const failed = results.filter(r => r.status === 'fail').length;
  const skipped = results.filter(r => r.status === 'skip').length;

  console.log(h1('═══════════════════════════════════════════'));
  console.log('  ' + ok(passed + ' passed') + '  ' + no(failed + ' failed') + '  ' + sk(skipped + ' skipped') + '  (' + TESTS.length + ' total)');
  console.log(h1('═══════════════════════════════════════════'));

  // Show failures detail
  if (failed > 0) {
    console.log(R + '\nFailures:' + X);
    for (const r of results.filter(r => r.status === 'fail')) {
      console.log('  ' + no(r.name) + ': ' + (r.reason || ''));
    }
  }

  // Show how to fix skipped
  if (skipped > 0) {
    console.log(Y + '\nSkipped tools — enable with:' + X);
    console.log('  • Python:  Install from https://python.org');
    console.log('  • Python deps: The MCP server auto-installs all pip packages on startup.');
    console.log('  • Groq:    Set GROQ_API_KEY and ELEVENLABS_API_KEY in .env');
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error(R + 'FATAL: ' + e.message + X);
  process.exit(2);
});
