/**
 * Quick smoke test — starts the MCP server and sends a tools/list request.
 * Run: npx tsx smoke-test.mjs
 */
import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const serverPath = resolve(__dirname, 'packages/server/src/index.ts');

console.log('Starting MCP server...');
const server = spawn('node', ['--import', 'tsx', serverPath], {
  cwd: __dirname,
  stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env },
});

let stdout = '';
let stderr = '';

server.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
server.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

server.on('error', (err) => {
  console.error('Failed to start server:', err.message);
  process.exit(1);
});

// Send tools/list request
const request = JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', id: 1 });
server.stdin.write(request + '\n');

// Wait for response
setTimeout(() => {
  console.log('=== STDOUT ===');
  console.log(stdout || '(no output)');
  console.log('=== STDERR ===');
  console.log(stderr || '(no output)');

  // Try to parse the response
  try {
    const lines = stdout.split('\n').filter(l => l.trim());
    for (const line of lines) {
      const parsed = JSON.parse(line);
      if (parsed.id === 1) {
        console.log('\n✅ SERVER RESPONDED!');
        if (parsed.result?.tools) {
          console.log(`Tools registered: ${parsed.result.tools.length}`);
          console.log('Tool names:', parsed.result.tools.map((t: any) => t.name).join(', '));
        }
      }
    }
  } catch {}

  server.kill();
  process.exit(0);
}, 10000);
