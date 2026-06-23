/**
 * Quick test — writes results to tests/result.txt
 */
import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SERVER_PATH = resolve(ROOT, 'packages/server/src/index.ts');

async function run() {
  const out = [];
  const log = (s) => { out.push(s); console.log(s); };

  const server = spawn('node', ['--import', 'tsx', SERVER_PATH], {
    cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env },
  });

  let stdout = '';
  server.stdout.on('data', (c) => { stdout += c.toString(); });

  await new Promise(r => setTimeout(r, 4000));
  log('Server started.\n');

  const tests = [
    ['mcp_health', {}],
    ['mcp_list_providers', {}],
    ['deeppipe_stats', {}],
    ['deeppipe_list_documents', {}],
    ['deeppipe_search', { query: 'test', limit: 3 }],
    ['deeppipe_ingest', { data: Buffer.from('Hello MCP test world. This is a test document.').toString('base64'), source: 'test.txt' }],
    ['deeppipe_get_document', { id: 1 }],
    ['deeppipe_get_text', { id: 1 }],
    ['deeppipe_extractive_answer', { question: 'What is this document?' }],
    ['deeppipe_chat_context', { question: 'What is this about?' }],
    ['piste_fact_check', { claim_text: 'The sky is blue because of Rayleigh scattering.', locale: 'en' }],
    ['piste_list_verdicts', {}],
    ['precis_list_documents', {}],
    ['precis_debug_stem', { q: 'payment terms' }],
    ['clinical_start_session', { lang: 'en' }],
    ['clinical_list_sessions', {}],
  ];

  let passed = 0, failed = 0, skipped = 0;

  for (let i = 0; i < tests.length; i++) {
    const [name, args] = tests[i];
    server.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: { name, arguments: args }, id: i + 1 }) + '\n');
    await new Promise(r => setTimeout(r, 2000));

    // Parse responses so far
    const lines = stdout.split('\n').filter(l => l.trim());
    let found = false;
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.id === i + 1) {
          found = true;
          if (parsed.error) {
            const msg = parsed.error.message || '';
            if (msg.includes('not reachable') || msg.includes('not configured') || msg.includes('NOT_FOUND')) {
              log(`  ⏭️  ${name}: skipped (${msg.slice(0,50)})`);
              skipped++;
            } else {
              log(`  ❌ ${name}: ${msg.slice(0,80)}`);
              failed++;
            }
          } else if (parsed.result?.isError) {
            log(`  ❌ ${name}: ${parsed.result.content?.[0]?.text?.slice(0,80) || 'error'}`);
            failed++;
          } else {
            log(`  ✅ ${name}: OK`);
            passed++;
          }
          break;
        }
      } catch {}
    }
    if (!found) {
      log(`  ❌ ${name}: no response`);
      failed++;
    }
  }

  server.kill();

  log(`\nResults: ${passed} passed, ${failed} failed, ${skipped} skipped (${tests.length} total)`);
  writeFileSync(resolve(__dirname, 'result.txt'), out.join('\n'));
}

run().catch(e => console.error('FATAL:', e));
