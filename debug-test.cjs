// debug-test.cjs — captures stderr, longer timeout
const { spawn } = require('child_process');
const { resolve } = require('path');
const fs = require('fs');

const ROOT = resolve(__dirname);
const SERVER = resolve(ROOT, 'packages/server/src/index.ts');
const OUT = resolve(ROOT, 'tests', 'debug-results.txt');

const log = [];
const add = (s) => { log.push(s); console.log(s); };

add('=== Starting MCP server ===');
add('Server path: ' + SERVER);
add('');

const proc = spawn('node', ['--import', 'tsx', SERVER], {
  cwd: ROOT,
  stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env },
});

let stdout = '';
let stderr = '';

proc.stdout.on('data', (c) => { stdout += c.toString(); });
proc.stderr.on('data', (c) => { 
  stderr += c.toString();
  // Log stderr immediately so we can see crash info
  process.stderr.write('[stderr] ' + c.toString());
});

proc.on('error', (err) => { add('SPAWN ERROR: ' + err.message); });
proc.on('exit', (code, sig) => { add('SERVER EXITED: code=' + code + ' signal=' + sig); });

// Wait for startup
setTimeout(() => {
  add('Server started (or crashed). Stdout length: ' + stdout.length + ', Stderr length: ' + stderr.length);
  
  if (stderr) {
    add('\n=== STDERR (first 2000 chars) ===');
    add(stderr.slice(0, 2000));
  }

  if (proc.killed) {
    add('\nServer already dead — cannot send requests.');
    add('\n=== FULL STDERR ===');
    add(stderr);
    fs.writeFileSync(OUT, log.join('\n'));
    process.exit(1);
  }

  add('\nSending tools/list request...');
  proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'tools/list', id: 1 }) + '\n');

  // Wait for response
  setTimeout(() => {
    add('\n=== RESPONSE CHECK ===');
    const lines = stdout.split('\n').filter(l => l.trim());
    add('Lines received: ' + lines.length);
    
    let found = false;
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.id === 1) {
          found = true;
          if (parsed.result?.tools) {
            add('SUCCESS! ' + parsed.result.tools.length + ' tools registered.');
            for (const t of parsed.result.tools) {
              add('  - ' + t.name);
            }
          } else if (parsed.error) {
            add('ERROR: ' + JSON.stringify(parsed.error));
          }
        }
      } catch {}
    }
    
    if (!found) {
      add('No tools/list response found in stdout.');
      add('\n=== FULL STDOUT ===');
      add(stdout || '(empty)');
      add('\n=== FULL STDERR ===');
      add(stderr || '(empty)');
    }

    proc.kill();
    fs.writeFileSync(OUT, log.join('\n'));
    add('\nResults saved to tests/debug-results.txt');
    process.exit(0);
  }, 15000);

}, 8000);
