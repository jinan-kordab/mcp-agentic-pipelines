/**
 * Bone-dry test — no integrations, no MCP SDK, just imports core and logs.
 * Run: npx tsx tests/bare-test.ts
 */
console.log('1. Starting bare test...');

// Test 1: can we import core?
console.log('2. Importing @unified-mcp/core...');
import { loadConfig, createLogger } from '@unified-mcp/core';
console.log('3. Core imported OK');

// Test 2: can we load config?
console.log('4. Loading config...');
const config = loadConfig();
console.log('5. Config loaded. Provider:', config.LLM_DEFAULT_PROVIDER);

// Test 3: can we import MCP SDK?
console.log('6. Importing MCP SDK...');
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
console.log('7. MCP SDK imported OK');

// Test 4: can we import DeepPipe engine?
console.log('8. Importing @kordabjinan/deeppipe...');
import { openPipeline } from '@kordabjinan/deeppipe';
console.log('9. DeepPipe engine imported OK');

// Test 5: can we open a pipeline?
console.log('10. Opening pipeline (in-memory)...');
const opened = openPipeline({ location: ':memory:' });
if (opened.ok) {
  console.log('11. Pipeline opened OK. Docs:', opened.value.documentCount());
  opened.value.close();
} else {
  console.log('11. Pipeline FAILED:', opened.error.message);
}

console.log('12. ALL TESTS PASSED');
