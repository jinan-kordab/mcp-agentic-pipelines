/**
 * Minimal MCP server — tests only what always works (no integrations).
 * Run: node --import tsx tests/minimal-server.ts
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { loadConfig, createLogger } from '@unified-mcp/core';

const config = loadConfig();
const logger = createLogger('debug');

logger.info('Minimal server starting...');

const server = new Server(
  { name: 'minimal-test', version: '0.0.0' },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{ name: 'ping', description: 'Returns pong.', inputSchema: { type: 'object', properties: {} } }],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name === 'ping') {
    return { content: [{ type: 'text', text: 'pong' }] };
  }
  return { content: [{ type: 'text', text: 'unknown' }], isError: true };
});

const transport = new StdioServerTransport();
server.connect(transport).then(() => {
  logger.info('Minimal server ready');
});

// Keep alive
process.stdin.resume();
