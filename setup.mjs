#!/usr/bin/env node
/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║     Unified MCP Server — Setup (uv-powered)                   ║
 * ║                                                              ║
 * ║  Uses uv (Astral) — the industry standard for MCP servers.   ║
 * ║  No Python pre-installed. No pip. No MAX_PATH issues.        ║
 * ║  One command:  node setup.mjs                                ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */
import { execSync } from 'node:child_process';
import { existsSync, readFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname);
const UV = resolve(ROOT, '.vendor', process.platform === 'win32' ? 'uv.exe' : 'uv');
const TARGET = resolve(ROOT, '.python-packages');
const ENV_FILE = resolve(ROOT, '.env');

// ANSI
const T='\x1b[32m✓\x1b[0m', X='\x1b[31m✗\x1b[0m', W='\x1b[33m⚠\x1b[0m';
const B='\x1b[1m', D='\x1b[2m', C='\x1b[36m', R='\x1b[0m';
const HR=D+'─'.repeat(55)+R;
let fail=false;
function head(t) { console.log(`\n${C}${B}  ${t}${R}\n  ${HR}`); }
function ok(m)  { console.log(`  ${T}  ${m}`); }
function err(m) { console.log(`  ${X}  ${m}`); fail=true; }
function warn(m){ console.log(`  ${W}  ${m}`); }
function info(m){ console.log(`     ${m}`); }
// execSync with shell — reliable on all platforms
function sh(cmd, opts={}) { return execSync(cmd, { cwd:ROOT, stdio:'pipe', encoding:'utf8', timeout:300_000, ...opts }); }

console.log(`\n${C}${B}  ╔══════════════════════════════════════════════════╗${R}`);
console.log(`${C}${B}  ║   MCP Agentic Pipelines — Setup (uv)             ║${R}`);
console.log(`${C}${B}  ║   5 integrations · 31 tools · One command            ║${R}`);
console.log(`${C}${B}  ╚══════════════════════════════════════════════════╝${R}`);

// 1. Node.js
head('Node.js Runtime');
const nodeV = process.version;
if (parseInt(nodeV.slice(1)) >= 18) ok(`Node.js ${nodeV}`);
else { err(`Need Node.js 18+. Found ${nodeV}`); process.exit(1); }

// 2. npm
head('Node.js Packages (npm)');
try { sh('node -e "require.resolve(\\"@unified-mcp/core\\")"'); ok('npm ready'); }
catch {
  info('Running npm install...');
  try { sh('npm install --prefer-offline', { stdio:'inherit' }); ok('npm complete'); }
  catch(e) { err(e.message); process.exit(1); }
}

// 3. uv
head('Package Manager (uv)');
if (!existsSync(UV)) {
  err('.vendor/uv.exe not found. Download it:');
  info('https://github.com/astral-sh/uv/releases/latest');
  info('Place uv.exe in .vendor/ and re-run.');
  process.exit(1);
}
ok(`uv ${sh(`"${UV}" --version`).trim()}`);

// 3b. Ensure proper Python (uv managed — no Store Python SSL issues)
head('Python Runtime');
const isStorePython = (() => {
  try {
    const pyPath = sh(`"${UV}" python find 3.11`).trim();
    return pyPath.includes('WindowsApps'); // Microsoft Store Python has broken SSL
  } catch { return false; }
})();

if (isStorePython) {
  warn('Microsoft Store Python detected — SSL is broken (sandboxed).');
  info('Installing uv-managed Python 3.11 (proper SSL, no sandbox)...');
  info('  This downloads ~25 MB — may take a minute on slow connections.');
  try {
    sh(`"${UV}" python install 3.11`, { stdio:'inherit', timeout: 180_000 });
    const managed = sh(`"${UV}" python find 3.11`).trim();
    ok(`uv-managed Python ready → ${managed}`);
  } catch(e) {
    warn(`uv-managed Python install failed: ${e.message.split('\n')[0]}`);
    info('Python tools (piste, precis) will be unavailable.');
    info('Fix: install python.org Python from https://python.org');
    info('Then re-run: node setup.mjs');
    // Non-fatal — npm-based tools still work
  }
} else {
  const pyPath = sh(`"${UV}" python find 3.11`).trim();
  ok(`Python 3.11 → ${pyPath}`);
}

// 4. Python packages
head('Python Packages (uv pip)');
mkdirSync(TARGET, { recursive: true });
const PKGS = ['dspy-ai','litellm','python-dotenv','fastapi','uvicorn','pydantic','numpy','nltk','sqlalchemy','httpx'];
const MODS = ['dspy','litellm','dotenv','fastapi','uvicorn','pydantic','numpy','nltk','sqlalchemy','httpx'];

info(`Installing ${PKGS.length} packages (uv is 10-100x faster than pip)...`);
try {
  sh(`"${UV}" pip install ${PKGS.join(' ')} --target "${TARGET}" --python 3.11`, { stdio:'inherit' });
  // Verify by checking file system (avoids Store Python SSL issues)
  let good = true;
  for (const mod of MODS) {
    if (existsSync(resolve(TARGET, mod)) || existsSync(resolve(TARGET, `${mod}.py`))) {
      ok(mod.padEnd(14));
    } else {
      // Some packages install under a different name (e.g. python-dotenv → dotenv)
      const altCheck = [mod, mod.replace(/-/g, '_'), `python_${mod}`];
      const found = altCheck.some(a => existsSync(resolve(TARGET, a)) || existsSync(resolve(TARGET, `${a}.py`)));
      if (found) ok(mod.padEnd(14));
      else { err(mod.padEnd(14)); good = false; }
    }
  }
  if (good) ok(`All ${PKGS.length} packages verified`);
} catch(e) {
  err(`uv install failed: ${e.message.split('\n')[0]}`);
}

if (fail) process.exit(1);

// 5. .env
head('Environment (.env)');
const KEYS = ['DEEPSEEK_API_KEY','OPENAI_API_KEY','GROQ_API_KEY','ELEVENLABS_API_KEY','TAVILY_API_KEY','SERPER_API_KEY','GOOGLE_CSE_API_KEY','GOOGLE_CSE_ID'];
if (existsSync(ENV_FILE)) {
  const env = readFileSync(ENV_FILE, 'utf8'); let n = 0;
  for (const k of KEYS) {
    if (new RegExp(`^${k}=.+`,'m').test(env)) { ok(k.padEnd(22)); n++; }
    else warn(`${k.padEnd(22)}${D}(optional)${R}`);
  }
  info(`\n  ${n}/${KEYS.length} keys configured`);
} else { warn('.env not found — create one with your API keys'); }

// Summary
console.log(`\n${HR}`);
if (fail) { console.log(`\n${X}  Setup failed. Fix and re-run: node setup.mjs\n`); process.exit(1); }
console.log(`\n${T}  ${B}Setup complete — all dependencies ready.${R}\n`);
console.log(`  ${B}Start server:${R}   ${C}npx mcp-agentic-pipelines${R}`);
console.log(`  ${B}Run tests:${R}      ${C}node test.mjs${R}`);
console.log(`  ${B}MCP config:${R}     ${D}{"command":"npx","args":["mcp-agentic-pipelines"]}${R}\n`);
