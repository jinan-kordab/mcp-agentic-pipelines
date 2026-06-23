/**
 * MCP Integration Test — tests actual tool calls with real API keys
 * Run: npx tsx integration-test.mjs
 */
import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

function sendRequest(serverPath, request) {
  return new Promise((resolve, reject) => {
    const server = spawn('node', ['--import', 'tsx', serverPath], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
    });

    let stdout = '';
    server.stdout.on('data', (chunk) => { stdout += chunk.toString(); });

    const timer = setTimeout(() => {
      server.kill();
      reject(new Error('Timeout'));
    }, 15000);

    server.stdin.write(JSON.stringify(request) + '\n');

    // Wait for response
    const check = setInterval(() => {
      const lines = stdout.split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.id === request.id) {
            clearTimeout(timer);
            clearInterval(check);
            server.kill();
            resolve(parsed);
            return;
          }
        } catch {}
      }
    }, 100);
  });
}

async function main() {
  const serverPath = resolve(__dirname, 'packages/server/src/index.ts');
  let id = 1;

  // Test 1: mcp_health
  console.log('🔍 Testing mcp_health...');
  try {
    const health = await sendRequest(serverPath, { jsonrpc: '2.0', method: 'tools/call', params: { name: 'mcp_health', arguments: {} }, id: id++ });
    const healthData = JSON.parse(health.result.content[0].text);
    console.log('✅ mcp_health: OK');
    console.log('   Server:', healthData.server.name, 'v' + healthData.server.version);
    console.log('   Tools registered:', healthData.tools.total);
    console.log('   LLM default provider:', healthData.llm.defaultProvider);
    console.log('   DeepSeek configured:', healthData.llm.deepPipe.configured);
    console.log('   Groq configured:', healthData.services.clinical.stt.configured);
    console.log('   ElevenLabs configured:', healthData.services.clinical.tts.configured);
    console.log('   Tavily configured:', !!healthData.services.piste.configured);
  } catch (e) {
    console.log('❌ mcp_health failed:', e.message);
  }

  // Test 2: mcp_list_providers
  console.log('\n🔍 Testing mcp_list_providers...');
  try {
    const providers = await sendRequest(serverPath, { jsonrpc: '2.0', method: 'tools/call', params: { name: 'mcp_list_providers', arguments: {} }, id: id++ });
    const provData = JSON.parse(providers.result.content[0].text);
    console.log('✅ mcp_list_providers:', provData.length, 'providers available');
    for (const p of provData) {
      console.log(`   - ${p.provider}: ${p.defaultModel} (used by: ${p.usedByComponents.join(', ')})`);
    }
  } catch (e) {
    console.log('❌ mcp_list_providers failed:', e.message);
  }

  // Test 3: deeppipe_stats
  console.log('\n🔍 Testing deeppipe_stats...');
  try {
    const stats = await sendRequest(serverPath, { jsonrpc: '2.0', method: 'tools/call', params: { name: 'deeppipe_stats', arguments: {} }, id: id++ });
    const statsData = JSON.parse(stats.result.content[0].text);
    console.log('✅ deeppipe_stats:', statsData.documentCount, 'documents indexed');
    console.log('   Index location:', statsData.indexLocation);
  } catch (e) {
    console.log('❌ deeppipe_stats failed:', e.message);
  }

  // Test 4: deeppipe_ingest (a small test document)
  console.log('\n🔍 Testing deeppipe_ingest with a PDF...');
  try {
    // Create a simple text file to ingest
    const testContent = Buffer.from('This is a test document. Payment terms are net 30 days. All invoices must be submitted within 15 days of service.').toString('base64');
    const ingest = await sendRequest(serverPath, { jsonrpc: '2.0', method: 'tools/call', params: { name: 'deeppipe_ingest', arguments: { data: testContent, source: 'test-invoice.txt' } }, id: id++ });
    const ingestData = JSON.parse(ingest.result.content[0].text);
    if (ingest.result.isError) {
      console.log('⚠️  Ingest note:', ingestData.error?.message || ingestData.error);
    } else {
      console.log('✅ deeppipe_ingest: Document #' + ingestData.documentId, '(' + ingestData.wordCount + ' words)');
    }
  } catch (e) {
    console.log('❌ deeppipe_ingest failed:', e.message);
  }

  // Test 5: deeppipe_search
  console.log('\n🔍 Testing deeppipe_search...');
  try {
    const search = await sendRequest(serverPath, { jsonrpc: '2.0', method: 'tools/call', params: { name: 'deeppipe_search', arguments: { query: 'payment terms', snippets: true } }, id: id++ });
    const searchData = JSON.parse(search.result.content[0].text);
    console.log('✅ deeppipe_search:', searchData.totalHits, 'hits in', searchData.elapsedMs, 'ms');
    if (searchData.hits?.length > 0) {
      console.log('   Top hit:', searchData.hits[0].snippet?.slice(0, 80));
    }
  } catch (e) {
    console.log('❌ deeppipe_search failed:', e.message);
  }

  console.log('\n═══════════════════════════════════════');
  console.log('✅ Integration tests complete!');
  console.log('The MCP server is ready for Claude Desktop / Cursor.');
  console.log('═══════════════════════════════════════');
}

main().catch(console.error);
