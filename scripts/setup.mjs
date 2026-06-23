#!/usr/bin/env node
/**
 * MCP Server Setup — one command to verify everything.
 * Run: node scripts/setup.mjs
 */

import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const PASS = '\x1b[32m✓\x1b[0m';
const FAIL = '\x1b[31m✗\x1b[0m';
const WARN = '\x1b[33m⚠\x1b[0m';

function check(label, fn) {
  try {
    const result = fn();
    console.log(`  ${PASS} ${label}${result ? ' — ' + result : ''}`);
    return true;
  } catch (e) {
    console.log(`  ${FAIL} ${label} — ${e.message}`);
    return false;
  }
}

console.log('\n╔══════════════════════════════════════╗');
console.log('║   Unified MCP Server — Setup Check   ║');
console.log('╚══════════════════════════════════════╝\n');

// ── Node.js ──────────────────────────────────────────────────────
console.log('── Runtime ──');
check('Node.js ' + process.version, () => process.version);
check('npm', () => execSync('npm --version', { encoding: 'utf8' }).trim());

// ── Dependencies ─────────────────────────────────────────────────
console.log('\n── Dependencies ──');
const nm = resolve(ROOT, 'node_modules');
check('node_modules/', () => existsSync(nm) ? 'exists' : (() => { throw new Error('Run: npm install'); })());
check('@modelcontextprotocol/sdk', () => existsSync(resolve(nm, '@modelcontextprotocol/sdk')) ? 'ok' : (() => { throw new Error('Run: npm install'); })());
check('@kordabjinan/deeppipe', () => existsSync(resolve(nm, '@kordabjinan/deeppipe')) ? 'ok' : (() => { throw new Error('Run: npm install'); })());
check('groq-sdk', () => existsSync(resolve(nm, 'groq-sdk')) ? 'ok' : (() => { throw new Error('Run: npm install'); })());
check('openai', () => existsSync(resolve(nm, 'openai')) ? 'ok' : (() => { throw new Error('Run: npm install'); })());
check('zod', () => existsSync(resolve(nm, 'zod')) ? 'ok' : (() => { throw new Error('Run: npm install'); })());

// ── Python ───────────────────────────────────────────────────────
console.log('\n── Python (for Piste + Precis) ──');
let hasPython = false;
for (const cmd of ['python', 'python3', 'py']) {
  try {
    const v = execSync(`"${cmd}" --version`, { encoding: 'utf8', timeout: 5000, windowsHide: true }).trim();
    if (v.toLowerCase().includes('python')) {
      console.log(`  ${PASS} ${cmd} — ${v}`);
      hasPython = true;
      break;
    }
  } catch {}
}
if (!hasPython) {
  console.log(`  ${WARN} Python not found. Piste/Precis tools will be unavailable.`);
  console.log('         Install from https://python.org');
}

// ── Python backends ──────────────────────────────────────────────
console.log('\n── Python Backend Files ──');
const pisteBridge = resolve(ROOT, '..', 'piste', 'bridge_piste.py');
const precisBridge = resolve(ROOT, '..', 'precis-agentic-pipeline', 'bridge_precis.py');
check('piste/bridge_piste.py', () => existsSync(pisteBridge) ? 'found' : (() => { throw new Error('Missing — clone piste repo'); })());
check('precis/bridge_precis.py', () => existsSync(precisBridge) ? 'found' : (() => { throw new Error('Missing — clone precis repo'); })());

// ── API Keys ─────────────────────────────────────────────────────
console.log('\n── API Keys (.env) ──');
const dotenv = (await import('dotenv')).default;
dotenv.config({ path: resolve(ROOT, '.env') });
const keys = ['LLM_DEFAULT_API_KEY', 'GROQ_API_KEY', 'ELEVENLABS_API_KEY', 'TAVILY_API_KEY'];
for (const k of keys) {
  const val = process.env[k];
  if (val && val.length > 4 && val !== 'sk-your-key-here') {
    console.log(`  ${PASS} ${k} = ${val.slice(0, 8)}...`);
  } else {
    console.log(`  ${WARN} ${k} not set — some tools will be limited`);
  }
}

// ── Integration files ────────────────────────────────────────────
console.log('\n── Integration Packages ──');
for (const pkg of ['core', 'deeppipe', 'piste', 'precis', 'clinical', 'server']) {
  const p = resolve(ROOT, 'packages', pkg, 'src', 'index.ts');
  check('packages/' + pkg, () => existsSync(p) ? 'ok' : (() => { throw new Error('Missing index.ts'); })());
}

console.log('\n╔══════════════════════════════════════╗');
console.log('║   Setup complete.                    ║');
console.log('║   Run: npm start                     ║');
console.log('║   Test: node fresh-test.cjs          ║');
console.log('╚══════════════════════════════════════╝\n');
